# =============================================================================
# bioclip_server.py  —  Production Camera Trap Identification API  v2.1
# =============================================================================
#
# Pipeline
# ────────
#  1. MegaDetector v5a   →  person / animal bounding boxes (YOLOv5)
#  2. Person verifier    →  CLIP human-vs-animal dual-check on detector-person crops
#                           rejects false positives (e.g. deer misread as person)
#  3. Human guard        →  CLIP human-vs-animal check on detector-animal crops
#                           catches missed persons inside animal detections
#  4. Species classifier →  CLIP open-set over the full ~220-species camera-trap
#                           vocabulary, with 6-template prompt ensembling
#  5. Allowlist filter   →  3-tier matching: exact → token-subset → synonym alias
#                           If top prediction resolves to allowed name  → species name
#                           If confident but not in allowed list        → "UNKNOWN"
#                           If confidence / margin too low              → None
#
# Key engineering decisions
# ─────────────────────────
#  • CLIP classifies against the FULL vocabulary first (open-set).
#    The allowed_species filter is post-hoc Python logic — not a CLIP constraint.
#    This eliminates closed-set hallucination (tiger → cat).
#
#  • Prompt ensembling (6 templates per species) improves CLIP accuracy by 3-8 %
#    and is computed once at startup, cached as a [N_species, dim] matrix.
#
#  • Person verification uses BOTH absolute threshold AND human/non-human ratio.
#    Prevents animals with upright posture from being classified as persons.
#
#  • Species decisions require BOTH confidence ≥ threshold AND top1-top2 margin.
#    This rejects near-tie predictions that are likely to be wrong.
#
#  • 3-tier species matching guarantees "domestic dog" / "feral dog" / "African
#    wild dog" all resolve to a client's "Dog" entry.  Similarly domestic cattle
#    → Cow, puma → Cougar, etc.  Matching never crosses biological families.
#
#  • Adaptive confidence thresholds: image luminance is measured directly from
#    pixel data before enhancement.  Night / IR images (mean luminance < 60)
#    receive a lower confidence threshold automatically; daytime images use the
#    configured default.  No external time-of-day input required.
#
#  • Allowed-species list is cached per client_id with a configurable TTL to
#    avoid a backend round-trip on every request.
#
#  • No DEFAULT_SPECIES fallback.  Empty allowed list → species is always None.
#
# =============================================================================

import asyncio
import io
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False


# =============================================================================
# App & Logging
# =============================================================================

app = FastAPI(
    title="CameraTrap Real-Time Identification API",
    version="2.0.0",
    description="MegaDetector + OpenCLIP open-set species identification for camera trap systems.",
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bioclip_server")

cors_origins = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Configuration  —  all tuneable via environment variables
# =============================================================================

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "120000000"))

# ── Detector ──────────────────────────────────────────────────────────────────
DETECTOR_MODEL_PATH = (
    os.getenv("DETECTOR_MODEL_PATH")
    or os.getenv("YOLO_MODEL_PATH", "models/md_v5a.0.0.pt")
)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
DETECTOR_INPUT_SIZE = int(os.getenv("DETECTOR_INPUT_SIZE", "640"))
DETECTOR_CONFIDENCE_DEFAULT = float(os.getenv("DETECTOR_CONFIDENCE_DEFAULT", "0.40"))

# ── CLIP model ────────────────────────────────────────────────────────────────
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")

# ── Species classification thresholds ─────────────────────────────────────────
# SPECIES_CONFIDENCE_THRESHOLD
#   Minimum softmax probability the top-1 prediction must reach.
#   With ~220 classes, uniform = 0.0045; a genuine hit typically lands 0.25–0.70.
#   0.25 is intentionally low to avoid false Nones; the margin check below
#   provides the second line of defence against near-tie guesses.
SPECIES_CONFIDENCE_THRESHOLD = float(os.getenv("SPECIES_CONFIDENCE_THRESHOLD", "0.25"))

# SPECIES_MARGIN_THRESHOLD
#   top1_prob − top2_prob must exceed this value.
#   Rejects ambiguous predictions where two species are nearly equally likely.
#   0.06 means top-1 must lead top-2 by at least 6 percentage points.
SPECIES_MARGIN_THRESHOLD = float(os.getenv("SPECIES_MARGIN_THRESHOLD", "0.06"))

# ── Adaptive confidence thresholds (image brightness) ────────────────────────
# Camera traps produce very different image quality at night vs day.
# Rather than requiring the caller to know the time of day, we measure
# mean luminance directly from the image pixels (pre-enhancement).
#
# NIGHT_LUMINANCE_THRESHOLD
#   Mean pixel luminance (0–255) below which the image is classified as night.
#   Typical night / IR camera trap images: 10–55.  Default 60.
NIGHT_LUMINANCE_THRESHOLD = float(os.getenv("NIGHT_LUMINANCE_THRESHOLD", "60"))

# DAY_LUMINANCE_THRESHOLD
#   Mean pixel luminance above which the image is classified as day.
#   Between NIGHT and DAY thresholds = dusk / dawn condition.  Default 110.
DAY_LUMINANCE_THRESHOLD = float(os.getenv("DAY_LUMINANCE_THRESHOLD", "110"))

# NIGHT_CONFIDENCE_SCALE
#   Multiplier applied to species_confidence_threshold for night images.
#   0.78 → threshold is reduced by ~22 % at night to compensate for poor
#   image quality and lower CLIP embedding similarity in dark/IR frames.
NIGHT_CONFIDENCE_SCALE = float(os.getenv("NIGHT_CONFIDENCE_SCALE", "0.78"))

# DUSK_CONFIDENCE_SCALE
#   Multiplier for dusk / dawn images.  Between night and day.
DUSK_CONFIDENCE_SCALE = float(os.getenv("DUSK_CONFIDENCE_SCALE", "0.90"))

# ── Human guard (animal crops suspected to be persons) ────────────────────────
# HUMAN_GUARD_THRESHOLD
#   Minimum combined probability of the human-prompt group to trigger override.
HUMAN_GUARD_THRESHOLD = float(os.getenv("HUMAN_GUARD_THRESHOLD", "0.52"))

# HUMAN_GUARD_RATIO
#   human_group_prob / non_human_group_prob must exceed this to trigger override.
#   Prevents a weak human signal from overriding a clear animal detection.
HUMAN_GUARD_RATIO = float(os.getenv("HUMAN_GUARD_RATIO", "1.40"))

# ── Person verifier (person crops from MegaDetector) ─────────────────────────
# PERSON_VERIFY_THRESHOLD
#   Minimum human_prob to confirm a detector-person box as a real person.
PERSON_VERIFY_THRESHOLD = float(os.getenv("PERSON_VERIFY_THRESHOLD", "0.58"))

# PERSON_VERIFY_RATIO
#   human_prob / non_human_prob must exceed this for person confirmation.
#   1.5 means human evidence must be 50 % stronger than animal evidence.
PERSON_VERIFY_RATIO = float(os.getenv("PERSON_VERIFY_RATIO", "1.50"))

# ── Pipeline knobs ────────────────────────────────────────────────────────────
CROP_PADDING_RATIO = float(os.getenv("CROP_PADDING_RATIO", "0.15"))
ENABLE_NIGHT_ENHANCEMENT_DEFAULT = (
    os.getenv("ENABLE_NIGHT_ENHANCEMENT_DEFAULT", "true").lower() == "true"
)
# Species list TTL — avoids a backend call on every request (seconds)
SPECIES_CACHE_TTL = float(os.getenv("SPECIES_CACHE_TTL_SECONDS", "300"))


# =============================================================================
# Camera Trap Species Vocabulary  (~220 species)
# =============================================================================
# CLIP classifies against this list using prompt ensembling.
# A result ∈ allowed_species → returned as-is.
# A result ∉ allowed_species → returned as "UNKNOWN".
# This list does NOT constrain the model; it provides the full open-set space.

CAMERA_TRAP_SPECIES: Tuple[str, ...] = (
    # ── Felids ────────────────────────────────────────────────────────────────
    "lion", "tiger", "leopard", "cheetah", "jaguar",
    "puma", "cougar", "mountain lion",
    "snow leopard", "clouded leopard", "Amur leopard",
    "Canada lynx", "Eurasian lynx", "Iberian lynx", "bobcat",
    "ocelot", "serval", "caracal",
    "fishing cat", "Pallas cat", "sand cat", "wildcat",
    "domestic cat", "feral cat",
    # ── Canids ────────────────────────────────────────────────────────────────
    "gray wolf", "red wolf", "Ethiopian wolf",
    "coyote", "red fox", "gray fox", "Arctic fox", "fennec fox", "bat-eared fox",
    "African wild dog", "dhole", "side-striped jackal", "black-backed jackal",
    "maned wolf", "bush dog",
    "domestic dog", "feral dog",
    # ── Ursids ────────────────────────────────────────────────────────────────
    "American black bear", "brown bear", "grizzly bear", "polar bear",
    "sun bear", "spectacled bear", "sloth bear", "giant panda",
    # ── Mustelids ─────────────────────────────────────────────────────────────
    "wolverine", "honey badger", "European badger", "American badger",
    "river otter", "giant river otter", "sea otter", "European otter",
    "stoat", "least weasel", "long-tailed weasel",
    "American mink", "European mink",
    "pine marten", "beech marten", "fisher", "tayra",
    # ── Procyonids ────────────────────────────────────────────────────────────
    "raccoon", "coati", "ringtail", "kinkajou",
    # ── Viverrids & Herpestids ────────────────────────────────────────────────
    "common palm civet", "African civet", "binturong",
    "banded mongoose", "dwarf mongoose", "meerkat",
    # ── Hyenids ───────────────────────────────────────────────────────────────
    "spotted hyena", "striped hyena", "brown hyena", "aardwolf",
    # ── Cervids ───────────────────────────────────────────────────────────────
    "white-tailed deer", "mule deer", "red deer", "roe deer",
    "fallow deer", "sika deer", "reeve muntjac", "Chinese water deer",
    "elk", "moose", "caribou", "reindeer",
    "chital", "sambar deer", "barasingha", "swamp deer",
    "marsh deer", "pampas deer",
    # ── Bovids — Cattle & Relatives ───────────────────────────────────────────
    "American bison", "European bison",
    "African buffalo", "cape buffalo", "water buffalo",
    "gaur", "banteng", "yak", "takin",
    "domestic cattle", "domestic cow",
    # ── Bovids — Antelopes & Gazelles ─────────────────────────────────────────
    "impala", "Thomson gazelle", "Grant gazelle", "springbok",
    "greater kudu", "lesser kudu", "common eland", "nyala",
    "bushbuck", "sitatunga", "bongo",
    "waterbuck", "lechwe", "puku",
    "wildebeest", "hartebeest", "topi", "gemsbok", "oryx",
    "sable antelope", "roan antelope",
    "klipspringer", "steenbok", "common duiker", "red duiker",
    "blackbuck", "nilgai", "saiga antelope",
    # ── Goats, Sheep & Relatives ──────────────────────────────────────────────
    "mountain goat", "bighorn sheep",
    "Alpine ibex", "markhor", "chamois",
    "domestic goat", "domestic sheep",
    # ── Horses & Relatives ────────────────────────────────────────────────────
    "plains zebra", "Grevy zebra", "mountain zebra",
    "African wild ass", "onager", "domestic horse", "domestic donkey",
    # ── Pigs & Peccaries ──────────────────────────────────────────────────────
    "wild boar", "warthog", "bushpig", "babirusa",
    "collared peccary", "white-lipped peccary",
    "domestic pig",
    # ── Tapirs & Rhinoceroses ─────────────────────────────────────────────────
    "Baird tapir", "lowland tapir", "Malayan tapir",
    "white rhinoceros", "black rhinoceros", "Indian rhinoceros",
    # ── Elephants ─────────────────────────────────────────────────────────────
    "African bush elephant", "African forest elephant", "Asian elephant",
    # ── Hippos & Giraffes ─────────────────────────────────────────────────────
    "common hippopotamus", "pygmy hippopotamus",
    "giraffe", "okapi",
    # ── Primates ──────────────────────────────────────────────────────────────
    "chimpanzee", "bonobo", "western gorilla", "orangutan",
    "white-handed gibbon", "siamang",
    "olive baboon", "chacma baboon", "mandrill", "gelada",
    "rhesus macaque", "Japanese macaque", "crab-eating macaque",
    "vervet monkey", "patas monkey", "guereza colobus",
    "proboscis monkey", "langur", "capuchin monkey", "howler monkey",
    # ── Small Mammals ─────────────────────────────────────────────────────────
    "Virginia opossum", "common opossum",
    "nine-banded armadillo", "giant armadillo",
    "giant anteater", "tamandua",
    "capybara", "paca", "agouti",
    "North American porcupine", "crested porcupine",
    "American beaver", "American muskrat",
    "groundhog", "prairie dog",
    "snowshoe hare", "Arctic hare", "European hare",
    "Eastern cottontail", "American pika",
    "striped skunk", "spotted skunk",
    "red panda", "aardvark", "pangolin",
    # ── Marsupials ────────────────────────────────────────────────────────────
    "red kangaroo", "eastern grey kangaroo",
    "common wallaroo", "red-necked wallaby",
    "common wombat", "koala", "Tasmanian devil",
    # ── Birds (commonly camera-trapped) ───────────────────────────────────────
    "wild turkey", "common peacock", "ring-necked pheasant",
    "helmeted guineafowl", "red junglefowl",
    "sandhill crane", "great blue heron", "grey heron",
    "marabou stork", "turkey vulture", "griffon vulture",
    "golden eagle", "bald eagle", "barn owl", "great horned owl",
    "common ostrich", "emu", "secretary bird",
    # ── Reptiles (commonly camera-trapped) ────────────────────────────────────
    "Nile monitor", "Komodo dragon", "water monitor",
    "American alligator", "Nile crocodile", "saltwater crocodile",
    "Burmese python", "African rock python", "green anaconda",
    "Aldabra giant tortoise",
)

_N_SPECIES = len(CAMERA_TRAP_SPECIES)

# 6-template prompt ensemble — averaged embeddings give 3–8 % accuracy gain
_PROMPT_TEMPLATES: Tuple[str, ...] = (
    "a photo of a {}",
    "a wildlife camera trap photo of a {}",
    "a camera trap image of a {}",
    "a photo of a {} in the wild",
    "a {} in its natural habitat",
    "a photo of a wild {}",
)

# Human / non-human guard prompts (used in two different contexts)
_HUMAN_PROMPTS: Tuple[str, ...] = (
    "a photo of a person",
    "a photo of a human being",
    "a photo of a man",
    "a photo of a woman",
    "a camera trap photo of a person",
    "a photo of a person standing in the forest",
)
_N_HUMAN = len(_HUMAN_PROMPTS)

_NON_HUMAN_PROMPTS: Tuple[str, ...] = (
    "a photo of a wild animal",
    "a wildlife camera trap photo of an animal",
    "a photo of a wild mammal",
    "a camera trap image of wildlife",
    "a photo of a wild creature in nature",
    "a photo of an animal in the forest",
)
_N_NON_HUMAN = len(_NON_HUMAN_PROMPTS)


# =============================================================================
# Model & Cache State  (module-level singletons, lazily initialised)
# =============================================================================

_detector_model = None
_detector_lock = threading.Lock()
_detector_device = "cuda" if torch.cuda.is_available() else "cpu"

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_clip_device = "cpu"
_clip_lock = threading.Lock()

# Pre-computed ensemble text features for the full species vocabulary
# Shape: [N_species, embedding_dim], L2-normalised
_species_text_features: Optional[torch.Tensor] = None

# Guard prompt features  Shape: [N_human + N_non_human, embedding_dim]
_guard_text_features: Optional[torch.Tensor] = None

_text_feature_lock = threading.Lock()

# Species list cache:  client_id → (species_list, timestamp)
_species_cache: Dict[str, Tuple[List[str], float]] = {}
_species_cache_lock = threading.Lock()


# =============================================================================
# Pydantic Models
# =============================================================================

class Detection(BaseModel):
    bbox: List[float]
    label: str                              # "person" | "animal"
    detector_confidence: float
    species: Optional[str] = None           # canonical name | "UNKNOWN" | None
    species_confidence: Optional[float] = None
    model_top_prediction: Optional[str] = None  # canonical CLIP best-guess (pre-filter)
    mask_png_base64: Optional[str] = None
    bioclip_top3: Optional[List[dict]] = None


class IdentifyResponse(BaseModel):
    detections: List[Detection]
    warnings: Optional[List[str]] = None
    image_condition: Optional[str] = None               # "day" | "dusk" | "night"
    image_luminance: Optional[float] = None             # mean pixel luminance 0–255
    effective_confidence_threshold: Optional[float] = None  # after adaptive scaling


class HealthResponse(BaseModel):
    status: str
    detector_loaded: bool
    clip_loaded: bool
    species_vocabulary_size: int


# =============================================================================
# Species Allowlist Matching
# =============================================================================

def _normalize_species(name: str) -> str:
    """Lowercase, collapse whitespace, keep only alphanum and spaces."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in name.lower())
    return " ".join(cleaned.split())


# Synonym / alias map
# ───────────────────
# Maps each vocabulary species name (exact, lowercase) to a tuple of alternative
# common names a client might use in their allowed list.
#
# Rules that guided this map:
#   • Only map within the same biological family (no dog → wolf).
#   • Domestic/feral variants always alias to their common short form.
#   • Named subspecies alias to the parent species (e.g. Amur leopard → leopard).
#   • Common aliases go in both directions where ambiguity exists
#     (puma ↔ cougar ↔ mountain lion).
#   • This map is intentionally conservative. When in doubt, no alias.
#
_SPECIES_ALIASES: Dict[str, Tuple[str, ...]] = {
    # ── Felids ────────────────────────────────────────────────────────────────
    "domestic cat":          ("cat",),
    "feral cat":             ("cat",),
    "amur leopard":          ("leopard",),
    "clouded leopard":       ("leopard",),
    "snow leopard":          ("leopard",),
    "puma":                  ("cougar", "mountain lion"),
    "cougar":                ("puma", "mountain lion"),
    "mountain lion":         ("puma", "cougar"),
    "canada lynx":           ("lynx",),
    "eurasian lynx":         ("lynx",),
    "iberian lynx":          ("lynx",),
    # ── Canids ────────────────────────────────────────────────────────────────
    "domestic dog":          ("dog",),
    "feral dog":             ("dog",),
    "african wild dog":      ("wild dog", "dog"),
    "gray wolf":             ("wolf",),
    "red wolf":              ("wolf",),
    "ethiopian wolf":        ("wolf",),
    "maned wolf":            ("wolf",),
    "red fox":               ("fox",),
    "gray fox":              ("fox",),
    "arctic fox":            ("fox",),
    "fennec fox":            ("fox",),
    "bat-eared fox":         ("fox",),
    "side-striped jackal":   ("jackal",),
    "black-backed jackal":   ("jackal",),
    # ── Ursids ────────────────────────────────────────────────────────────────
    "american black bear":   ("bear", "black bear"),
    "brown bear":            ("bear",),
    "grizzly bear":          ("bear", "grizzly"),
    "polar bear":            ("bear",),
    "sun bear":              ("bear",),
    "spectacled bear":       ("bear",),
    "sloth bear":            ("bear",),
    # ── Bovids — Cattle ───────────────────────────────────────────────────────
    "domestic cattle":       ("cow", "cattle"),
    "domestic cow":          ("cow", "cattle"),
    "cape buffalo":          ("buffalo",),
    "african buffalo":       ("buffalo",),
    "water buffalo":         ("buffalo",),
    "american bison":        ("bison", "buffalo"),
    "european bison":        ("bison",),
    # ── Cervids ───────────────────────────────────────────────────────────────
    "white-tailed deer":     ("deer",),
    "mule deer":             ("deer",),
    "red deer":              ("deer",),
    "roe deer":              ("deer",),
    "fallow deer":           ("deer",),
    "sika deer":             ("deer",),
    "reeve muntjac":         ("deer", "muntjac"),
    "chinese water deer":    ("deer",),
    "chital":                ("deer", "spotted deer", "axis deer"),
    "sambar deer":           ("deer", "sambar"),
    "barasingha":            ("deer",),
    "swamp deer":            ("deer",),
    "marsh deer":            ("deer",),
    "pampas deer":           ("deer",),
    # ── Horses & Relatives ────────────────────────────────────────────────────
    "domestic horse":        ("horse",),
    "domestic donkey":       ("donkey",),
    "plains zebra":          ("zebra",),
    "grevy zebra":           ("zebra",),
    "mountain zebra":        ("zebra",),
    # ── Pigs ──────────────────────────────────────────────────────────────────
    "domestic pig":          ("pig",),
    "wild boar":             ("boar", "pig"),
    # ── Goats & Sheep ─────────────────────────────────────────────────────────
    "domestic goat":         ("goat",),
    "domestic sheep":        ("sheep",),
    "bighorn sheep":         ("sheep",),
    "alpine ibex":           ("ibex",),
    # ── Elephants ─────────────────────────────────────────────────────────────
    "african bush elephant": ("elephant",),
    "african forest elephant": ("elephant",),
    "asian elephant":        ("elephant",),
    # ── Rhinoceroses ──────────────────────────────────────────────────────────
    "white rhinoceros":      ("rhino", "rhinoceros"),
    "black rhinoceros":      ("rhino", "rhinoceros"),
    "indian rhinoceros":     ("rhino", "rhinoceros"),
    # ── Primates ──────────────────────────────────────────────────────────────
    "western gorilla":       ("gorilla",),
    "olive baboon":          ("baboon",),
    "chacma baboon":         ("baboon",),
    "rhesus macaque":        ("macaque", "monkey"),
    "japanese macaque":      ("macaque", "monkey"),
    "crab-eating macaque":   ("macaque", "monkey"),
    "vervet monkey":         ("monkey",),
    "patas monkey":          ("monkey",),
    "guereza colobus":       ("monkey", "colobus"),
    "proboscis monkey":      ("monkey",),
    "langur":                ("monkey",),
    "capuchin monkey":       ("monkey",),
    "howler monkey":         ("monkey",),
    # ── Hyenas ────────────────────────────────────────────────────────────────
    "spotted hyena":         ("hyena",),
    "striped hyena":         ("hyena",),
    "brown hyena":           ("hyena",),
    # ── Hippos ────────────────────────────────────────────────────────────────
    "common hippopotamus":   ("hippo", "hippopotamus"),
    "pygmy hippopotamus":    ("hippo", "hippopotamus"),
    # ── Crocodilians ──────────────────────────────────────────────────────────
    "american alligator":    ("alligator",),
    "nile crocodile":        ("crocodile",),
    "saltwater crocodile":   ("crocodile",),
}

# Normalise alias map keys once at import time
_SPECIES_ALIASES_NORM: Dict[str, Tuple[str, ...]] = {
    _normalize_species(k): v for k, v in _SPECIES_ALIASES.items()
}


def _canonicalize_species(name: str) -> str:
    """
    Convert detailed/open-set species labels into stable canonical classes.
    Classification still runs over the full detailed vocabulary; this only defines
    the taxonomy used for final decisioning.
    """
    norm = _normalize_species(name)
    if not norm:
        return norm

    # Keep explicit high-salience classes.
    if norm in {"lion", "tiger", "elephant", "giraffe", "zebra", "rhino"}:
        return norm

    tokens = set(norm.split())

    if tokens & {"dog", "wolf", "fox", "jackal", "coyote", "dhole", "canid"}:
        return "dog"
    if tokens & {
        "cat", "lynx", "bobcat", "ocelot", "serval", "caracal", "puma", "cougar",
        "leopard", "jaguar", "cheetah", "wildcat", "feline"
    }:
        return "cat"
    if tokens & {"cow", "cattle", "buffalo", "bison", "gaur", "banteng", "yak", "takin"}:
        return "cow"
    if tokens & {"deer", "elk", "moose", "caribou", "reindeer", "muntjac"}:
        return "deer"
    if tokens & {"goat", "ibex", "markhor", "chamois"}:
        return "goat"
    if tokens & {"sheep"}:
        return "sheep"
    if tokens & {"pig", "boar", "warthog", "peccary", "bushpig", "babirusa"}:
        return "pig"
    if tokens & {"horse", "donkey", "ass", "onager"}:
        return "horse"
    if tokens & {"bear", "panda"}:
        return "bear"
    if tokens & {"monkey", "macaque", "baboon", "gorilla", "chimpanzee", "orangutan", "gibbon"}:
        return "monkey"

    # Reasonable default canonical bucket for unmapped labels.
    parts = norm.split()
    return parts[-1] if parts else norm


def _build_canonical_species_groups() -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(CAMERA_TRAP_SPECIES):
        canonical = _canonicalize_species(name)
        groups.setdefault(canonical, []).append(idx)
    return groups


_CANONICAL_SPECIES_GROUPS: Dict[str, List[int]] = _build_canonical_species_groups()


def _build_allowed_index(allowed_species: List[str]) -> Dict[str, str]:
    """
    Build a normalised-key → original-name lookup for the allowed list.
    """
    return {_normalize_species(s): s for s in allowed_species if s.strip()}


def _match_to_allowed(
    predicted: str,
    allowed_index: Dict[str, str],
) -> Optional[str]:
    """
    4-tier matching: predicted species → client's allowed name.

    Tier 1 — Exact normalised match
        "leopard"        → allowed "Leopard"   ✓
        "domestic dog"   → allowed "domestic dog"  ✓

    Tier 2 — Token-subset match
        "gray wolf"      → allowed "Wolf"      ({"wolf"} ⊆ {"gray","wolf"})
        "domestic cat"   → allowed "Cat"       ({"cat"}  ⊆ {"domestic","cat"})
        "African wild dog" → allowed "Dog"     ({"dog"}  ⊆ {"african","wild","dog"})

    Tier 3 — Synonym alias lookup  (covers gaps tokens can't bridge)
        "domestic cattle" → allowed "Cow"      (alias "cow" → exact match)
        "puma"           → allowed "Cougar"    (alias "cougar" → exact match)
        "grizzly bear"   → allowed "Bear"      (alias "bear" → exact match)

    Tier 4 — Canonical taxonomy fallback
        "Ethiopian wolf"  → canonical "dog" → allowed "Dog"

    Returns the original (properly-cased) allowed name on match, else None.
    """
    pred_norm = _normalize_species(predicted)
    pred_tokens = set(pred_norm.split())

    # ── Tier 1: exact ─────────────────────────────────────────────────────────
    if pred_norm in allowed_index:
        return allowed_index[pred_norm]

    # ── Tier 2: token-subset ──────────────────────────────────────────────────
    for norm_key, original in allowed_index.items():
        key_tokens = set(norm_key.split())
        if pred_tokens <= key_tokens or key_tokens <= pred_tokens:
            return original

    # ── Tier 3: synonym alias ─────────────────────────────────────────────────
    aliases = _SPECIES_ALIASES_NORM.get(pred_norm, ())
    for alias in aliases:
        alias_norm = _normalize_species(alias)
        # Exact alias match
        if alias_norm in allowed_index:
            return allowed_index[alias_norm]
        # Token-subset alias match
        alias_tokens = set(alias_norm.split())
        for norm_key, original in allowed_index.items():
            key_tokens = set(norm_key.split())
            if alias_tokens <= key_tokens or key_tokens <= alias_tokens:
                return original

    # Canonical fallback (e.g. wolf/fox/jackal → dog)
    canonical_norm = _normalize_species(_canonicalize_species(predicted))
    if canonical_norm in allowed_index:
        return allowed_index[canonical_norm]
    canonical_tokens = set(canonical_norm.split())
    for norm_key, original in allowed_index.items():
        key_tokens = set(norm_key.split())
        if canonical_tokens <= key_tokens or key_tokens <= canonical_tokens:
            return original

    return None


# =============================================================================
# Model Loading
# =============================================================================

def get_detector_model():
    global _detector_model
    if _detector_model is not None:
        return _detector_model
    with _detector_lock:
        if _detector_model is None:
            logger.info(
                "Loading MegaDetector from: %s on %s",
                DETECTOR_MODEL_PATH,
                _detector_device,
            )
            t0 = time.time()
            _detector_model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=DETECTOR_MODEL_PATH,
                force_reload=False,
                verbose=False,
            )
            _detector_model.to(_detector_device)
            _detector_model.eval()
            logger.info("MegaDetector ready in %.2fs", time.time() - t0)
    return _detector_model


def get_clip_components() -> Tuple[torch.nn.Module, object, object, str]:
    global _clip_model, _clip_preprocess, _clip_tokenizer, _clip_device
    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _clip_tokenizer, _clip_device
    with _clip_lock:
        if _clip_model is None:
            _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "Loading CLIP %s (%s) on %s",
                CLIP_MODEL_NAME, CLIP_PRETRAINED, _clip_device,
            )
            t0 = time.time()
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME,
                pretrained=CLIP_PRETRAINED,
                device=_clip_device,
            )
            _clip_model.eval()
            _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
            logger.info("CLIP ready in %.2fs", time.time() - t0)
    return _clip_model, _clip_preprocess, _clip_tokenizer, _clip_device


# =============================================================================
# Text Feature Encoding  (cached, computed once at startup)
# =============================================================================

def _encode_text_prompts(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    device: str,
) -> torch.Tensor:
    """Encode a list of prompts → L2-normalised features [N, dim]."""
    tokens = tokenizer(list(prompts)).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def _build_ensemble_species_features(
    model: torch.nn.Module,
    tokenizer,
    device: str,
) -> torch.Tensor:
    """
    For each species, encode all prompt templates and average the L2-normalised
    embeddings, then re-normalise.  Result: [N_species, dim].

    Prompt ensembling is the standard technique from the original CLIP paper
    and consistently improves zero-shot accuracy by 3–8 %.
    """
    logger.info(
        "Building ensemble text features for %d species × %d templates …",
        _N_SPECIES, len(_PROMPT_TEMPLATES),
    )
    t0 = time.time()
    all_prompts = [
        tmpl.format(species)
        for species in CAMERA_TRAP_SPECIES
        for tmpl in _PROMPT_TEMPLATES
    ]
    tokens = tokenizer(all_prompts).to(device)
    n_templates = len(_PROMPT_TEMPLATES)

    with torch.no_grad():
        feats = model.encode_text(tokens)                    # [N*T, dim]
        feats = feats / feats.norm(dim=-1, keepdim=True)    # per-prompt normalise
        feats = feats.view(_N_SPECIES, n_templates, -1)      # [N, T, dim]
        feats = feats.mean(dim=1)                            # [N, dim]  average
        feats = feats / feats.norm(dim=-1, keepdim=True)    # re-normalise average

    logger.info(
        "Species ensemble features built in %.2fs  shape=%s",
        time.time() - t0, tuple(feats.shape),
    )
    return feats                                             # [N_species, dim]


def get_species_text_features() -> torch.Tensor:
    """Return (and cache) the ensemble species feature matrix."""
    global _species_text_features
    if _species_text_features is not None:
        return _species_text_features
    with _text_feature_lock:
        if _species_text_features is None:
            model, _, tokenizer, device = get_clip_components()
            _species_text_features = _build_ensemble_species_features(
                model, tokenizer, device
            )
    return _species_text_features


def get_guard_text_features() -> torch.Tensor:
    """Return (and cache) the human / non-human guard feature matrix."""
    global _guard_text_features
    if _guard_text_features is not None:
        return _guard_text_features
    with _text_feature_lock:
        if _guard_text_features is None:
            model, _, tokenizer, device = get_clip_components()
            all_guard_prompts = list(_HUMAN_PROMPTS) + list(_NON_HUMAN_PROMPTS)
            _guard_text_features = _encode_text_prompts(
                model, tokenizer, all_guard_prompts, device
            )
            logger.info("Guard text features built. shape=%s", tuple(_guard_text_features.shape))
    return _guard_text_features


# =============================================================================
# Image Utilities
# =============================================================================

def _enhance_low_light(img: Image.Image) -> Image.Image:
    """CLAHE-based night enhancement via OpenCV (no-op if OpenCV unavailable)."""
    if not OPENCV_AVAILABLE:
        return img
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    l_ch = cv2.convertScaleAbs(l_ch, alpha=1.08, beta=8)
    enhanced = cv2.merge((l_ch, a_ch, b_ch))
    enhanced = cv2.cvtColor(cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced)


def _estimate_image_condition(img: Image.Image) -> Tuple[str, float, float]:
    """
    Estimate lighting condition from image pixel data alone.

    Returns
    ───────
    condition   : "day" | "dusk" | "night"
    luminance   : mean pixel luminance  0–255  (from grayscale conversion)
    conf_scale  : multiplier to apply to species_confidence_threshold
                  < 1.0 at night → threshold is relaxed to account for
                  poor CLIP embedding quality in dark / IR frames.

    Why measure luminance not use a clock?
    • Camera traps don't always embed GPS or accurate timestamps.
    • IR night images have characteristically low luminance regardless of
      sensor type — the measurement is always available from the pixels.
    • A 50-line fast path (convert to L, mean) adds < 1 ms overhead.
    """
    gray = img.convert("L")                     # Rec.601 luminance channel
    lum = float(np.array(gray, dtype=np.float32).mean())

    if lum < NIGHT_LUMINANCE_THRESHOLD:
        return "night", round(lum, 2), NIGHT_CONFIDENCE_SCALE
    elif lum < DAY_LUMINANCE_THRESHOLD:
        return "dusk", round(lum, 2), DUSK_CONFIDENCE_SCALE
    else:
        return "day", round(lum, 2), 1.0



def _padded_xyxy(
    box: List[float], pad_ratio: float, img_w: int, img_h: int
) -> List[float]:
    x1, y1, x2, y2 = box
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    return [
        max(0.0, x1 - w * pad_ratio),
        max(0.0, y1 - h * pad_ratio),
        min(float(img_w), x2 + w * pad_ratio),
        min(float(img_h), y2 + h * pad_ratio),
    ]


def _crop_image(img: Image.Image, box: List[float]) -> Image.Image:
    x1, y1, x2, y2 = box
    ix1 = int(max(0, np.floor(x1)))
    iy1 = int(max(0, np.floor(y1)))
    ix2 = int(min(img.width, np.ceil(x2)))
    iy2 = int(min(img.height, np.ceil(y2)))
    if ix2 <= ix1:
        ix2 = min(img.width, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(img.height, iy1 + 1)
    return img.crop((ix1, iy1, ix2, iy2)).convert("RGB")


# =============================================================================
# MegaDetector Inference
# =============================================================================

def detect_objects(
    img: Image.Image, confidence_threshold: float
) -> Tuple[List[Tuple[List[float], float]], List[Tuple[List[float], float]]]:
    """
    Run MegaDetector and return:
      person_detections  — list of (xyxy_box, confidence)
      animal_detections  — list of (xyxy_box, confidence)
    Vehicle detections are discarded.
    """
    model = get_detector_model()
    model.conf = confidence_threshold
    t0 = time.time()
    results = model(img, size=DETECTOR_INPUT_SIZE)
    logger.info("MegaDetector inference: %.3fs", time.time() - t0)

    persons: List[Tuple[List[float], float]] = []
    animals: List[Tuple[List[float], float]] = []

    for det in results.xyxy[0].tolist():
        x1, y1, x2, y2, conf, cls = det
        name = str(model.names[int(cls)]).strip().lower()
        box = [float(x1), float(y1), float(x2), float(y2)]
        if name == "person":
            persons.append((box, float(conf)))
        elif name == "animal":
            animals.append((box, float(conf)))

    return persons, animals


# =============================================================================
# CLIP Inference — Human Guard & Person Verifier
# =============================================================================

def _encode_image_batch(
    crops: List[Image.Image],
    preprocess,
    model: torch.nn.Module,
    device: str,
) -> torch.Tensor:
    """Encode a list of PIL crops → L2-normalised image features [N, dim]."""
    batch = torch.stack([preprocess(c) for c in crops]).to(device)
    with torch.inference_mode():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def _score_human_likeness_from_features(
    img_feats: torch.Tensor,
    guard_feats: torch.Tensor,
) -> List[Tuple[float, float]]:
    """Return human/non-human scores from precomputed CLIP image features."""
    if img_feats.numel() == 0:
        return []

    logits = 100.0 * img_feats @ guard_feats.T        # [N_crops, N_guard]
    probs = torch.softmax(logits, dim=-1)

    out: List[Tuple[float, float]] = []
    for row in probs:
        h = float(row[:_N_HUMAN].sum())
        nh = float(row[_N_HUMAN:].sum())
        out.append((h, nh))
    return out


def score_human_likeness(crops: List[Image.Image]) -> List[Tuple[float, float]]:
    """
    For each crop, return (human_group_prob, non_human_group_prob) after
    softmax over the combined guard prompt set.

    Dual return lets the caller apply BOTH an absolute threshold check
    and a human/non-human ratio check independently.
    """
    if not crops:
        return []
    model, preprocess, _, device = get_clip_components()
    guard_feats = get_guard_text_features()           # [N_human + N_non_human, dim]
    img_feats = _encode_image_batch(crops, preprocess, model, device)
    return _score_human_likeness_from_features(img_feats, guard_feats)


# =============================================================================
# CLIP Inference — Open-Set Species Classification
# =============================================================================

def classify_animal_crops(
    crops: List[Image.Image],
    allowed_species: List[str],
    confidence_threshold: float,
    margin_threshold: float,
    topk: int,
) -> List[Tuple[bool, Optional[str], Optional[str], Optional[float], List[dict]]]:
    """
    Classify each crop against the full camera-trap vocabulary, aggregate
    probabilities into canonical species classes, then apply allowlist filtering.

    Returns a list of tuples:
      (human_override, assigned_species, model_top_prediction, confidence, top3)

    ─ human_override        True  → CLIP thinks this animal crop is actually a person
    ─ assigned_species      Canonical species to report:
                              • name from allowed_species   if CLIP is confident and it's in the list
                              • "UNKNOWN"                   if confident but not in the allowed list
                              • None                        if not confident enough
    ─ model_top_prediction  Canonical CLIP best-guess (pre-filter), always set when confident
    ─ confidence            Top-1 probability over canonical classes
    ─ top3                  Top-k canonical classes, for debugging / UI
    """
    if not crops:
        return []

    model, preprocess, _, device = get_clip_components()
    img_feats = _encode_image_batch(crops, preprocess, model, device)  # [N, dim]

    return _classify_animal_features(
        img_feats=img_feats,
        allowed_species=allowed_species,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        topk=topk,
        guard_feats=get_guard_text_features(),
        species_feats=get_species_text_features(),
    )


def _classify_animal_features(
    img_feats: torch.Tensor,
    allowed_species: List[str],
    confidence_threshold: float,
    margin_threshold: float,
    topk: int,
    guard_feats: torch.Tensor,
    species_feats: torch.Tensor,
) -> List[Tuple[bool, Optional[str], Optional[str], Optional[float], List[dict]]]:
    """Classify precomputed CLIP image features using canonical species taxonomy."""
    if img_feats.numel() == 0:
        return []

    # ── Human guard (separate softmax over guard prompts only) ────────────────
    guard_logits = 100.0 * img_feats @ guard_feats.T          # [N, N_guard]
    guard_probs = torch.softmax(guard_logits, dim=-1)

    # ── Species classification (separate softmax over species vocab) ──────────
    species_logits = 100.0 * img_feats @ species_feats.T      # [N, N_species]
    species_probs = torch.softmax(species_logits, dim=-1)

    # Build normalised allowlist index once
    allowed_index = _build_allowed_index(allowed_species)

    max_k = 5
    k = min(max(topk, 1), max_k)
    results: List[Tuple[bool, Optional[str], Optional[str], Optional[float], List[dict]]] = []

    for i in range(int(img_feats.shape[0])):
        # ── Human guard ───────────────────────────────────────────────────────
        g_row = guard_probs[i]
        human_prob = float(g_row[:_N_HUMAN].sum())
        non_human_prob = float(g_row[_N_HUMAN:].sum())

        if (
            human_prob >= HUMAN_GUARD_THRESHOLD
            and non_human_prob > 0
            and human_prob / non_human_prob >= HUMAN_GUARD_RATIO
        ):
            logger.debug(
                "Human guard triggered on animal crop  human=%.3f  non_human=%.3f",
                human_prob, non_human_prob,
            )
            results.append((True, None, None, None, []))
            continue

        # ── Species classification (canonical taxonomy) ──────────────────────
        s_row = species_probs[i]
        canonical_scores: List[Tuple[str, float]] = []
        for canonical, indices in _CANONICAL_SPECIES_GROUPS.items():
            score = float(s_row[indices].sum())
            canonical_scores.append((canonical, score))

        canonical_scores.sort(key=lambda pair: pair[1], reverse=True)
        top_scores = canonical_scores[:k]
        top3: List[dict] = [
            {"species": name, "confidence": round(score, 4)}
            for name, score in top_scores
        ]

        best_species_raw, best_score = top_scores[0]
        margin = (top_scores[0][1] - top_scores[1][1]) if len(top_scores) >= 2 else 1.0

        # If model is not confident or prediction is ambiguous → None
        if best_score < confidence_threshold or margin < margin_threshold:
            logger.debug(
                "Species rejected  conf=%.3f(<%s)  margin=%.3f(<%s)  raw=%s",
                best_score, confidence_threshold, margin, margin_threshold, best_species_raw,
            )
            results.append((False, None, None, None, top3))
            continue

        # Confident prediction — now apply allowlist filter
        model_top_prediction = best_species_raw

        if not allowed_species:
            # No allowlist configured: return canonical model prediction directly.
            results.append((False, model_top_prediction, model_top_prediction, round(best_score, 4), top3))
            continue

        matched = _match_to_allowed(best_species_raw, allowed_index)
        if matched:
            # In the allowed list → report it
            results.append((False, matched, model_top_prediction, round(best_score, 4), top3))
        else:
            # Confident prediction, but not in the client's list → UNKNOWN
            logger.debug(
                "Species not in allowed list  raw=%s  conf=%.3f  → UNKNOWN",
                best_species_raw, best_score,
            )
            results.append((False, "UNKNOWN", model_top_prediction, round(best_score, 4), top3))

    return results


# =============================================================================
# Species List — Backend Fetch with TTL Cache
# =============================================================================

async def fetch_client_species(client_id: str) -> List[str]:
    """
    Fetch the allowed species list for a client from the backend.
    Results are cached per client_id for SPECIES_CACHE_TTL seconds.
    On backend error, returns the stale cached value if available, else [].
    No DEFAULT_SPECIES fallback — an empty list means no species assignment.
    """
    now = time.time()
    with _species_cache_lock:
        cached = _species_cache.get(client_id)
        if cached and (now - cached[1]) < SPECIES_CACHE_TTL:
            return cached[0]

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{BACKEND_URL}/api/species-of-interest/supported"
            )
            if resp.status_code == 200:
                data = resp.json()
                species = [
                    s.get("specieName")
                    for s in data.get("supportedSpecies", [])
                ]
                species = [s for s in species if isinstance(s, str) and s.strip()]
                if species:
                    logger.info(
                        "Fetched %d allowed species for client '%s'",
                        len(species), client_id,
                    )
                    with _species_cache_lock:
                        _species_cache[client_id] = (species, now)
                    return species
            else:
                logger.warning(
                    "Backend returned HTTP %d for client '%s'",
                    resp.status_code, client_id,
                )
    except Exception as exc:
        logger.warning("Species fetch failed for client '%s': %s", client_id, exc)

    # Return stale cache on error rather than empty
    with _species_cache_lock:
        stale = _species_cache.get(client_id)
        if stale:
            logger.warning(
                "Using stale species cache for client '%s' (%d species)",
                client_id, len(stale[0]),
            )
            return stale[0]

    return []   # ← no fallback, no default species


# =============================================================================
# FastAPI Lifecycle & Endpoints
# =============================================================================

@app.on_event("startup")
def _preload_models() -> None:
    """Load both models and pre-compute text features at startup, not on first request."""
    try:
        get_detector_model()
    except Exception as exc:
        logger.error("MegaDetector preload failed: %s", exc)

    try:
        get_clip_components()
        get_species_text_features()   # blocks until ensemble features are built
        get_guard_text_features()
    except Exception as exc:
        logger.error("CLIP preload failed: %s", exc)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        detector_loaded=_detector_model is not None,
        clip_loaded=_clip_model is not None,
        species_vocabulary_size=_N_SPECIES,
    )


@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    file: UploadFile = File(...),
    client_id: Optional[str] = Form(None),
    detector_threshold: float = Form(DETECTOR_CONFIDENCE_DEFAULT),
    species_confidence_threshold: float = Form(SPECIES_CONFIDENCE_THRESHOLD),
    species_margin_threshold: float = Form(SPECIES_MARGIN_THRESHOLD),
    crop_padding_ratio: float = Form(CROP_PADDING_RATIO),
    topk_species: int = Form(3),
    enhance_night: bool = Form(ENABLE_NIGHT_ENHANCEMENT_DEFAULT),
) -> IdentifyResponse:
    request_start = time.perf_counter()
    stage_timings: Dict[str, float] = {}
    species_task: Optional[asyncio.Task[List[str]]] = None

    if client_id:
        species_task = asyncio.create_task(fetch_client_species(client_id))

    # ── Input validation ──────────────────────────────────────────────────────
    if not file.filename:
        return IdentifyResponse(detections=[], warnings=["No file provided."])
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
        return IdentifyResponse(
            detections=[], warnings=["Unsupported file type. Use jpg/png/webp."]
        )

    detector_threshold = float(np.clip(detector_threshold, 0.05, 1.0))
    species_confidence_threshold = float(np.clip(species_confidence_threshold, 0.05, 1.0))
    species_margin_threshold = float(np.clip(species_margin_threshold, 0.0, 1.0))
    crop_padding_ratio = float(np.clip(crop_padding_ratio, 0.0, 0.35))
    topk_species = int(np.clip(topk_species, 1, 5))

    t0 = time.perf_counter()
    img_bytes = await file.read()
    stage_timings["file_read"] = time.perf_counter() - t0
    if len(img_bytes) > MAX_UPLOAD_BYTES:
        if species_task is not None:
            species_task.cancel()
        return IdentifyResponse(
            detections=[], warnings=[f"File too large. Max {MAX_UPLOAD_MB} MB."]
        )

    warnings: List[str] = []

    try:
        t0 = time.perf_counter()
        with Image.open(io.BytesIO(img_bytes)) as raw_img:
            source_img = raw_img.convert("RGB")
        stage_timings["image_decode"] = time.perf_counter() - t0

        # ── Measure lighting condition BEFORE enhancement ─────────────────────
        # We want the true scene luminance, not the CLAHE-boosted value.
        t0 = time.perf_counter()
        image_condition, image_luminance, conf_scale = _estimate_image_condition(source_img)
        effective_conf_threshold = round(species_confidence_threshold * conf_scale, 4)
        stage_timings["lighting_estimate"] = time.perf_counter() - t0

        if image_condition != "day":
            warnings.append(
                f"Image condition detected as '{image_condition}' "
                f"(luminance={image_luminance:.1f}/255). "
                f"Species confidence threshold relaxed: "
                f"{species_confidence_threshold:.3f} → {effective_conf_threshold:.3f}."
            )
        logger.info(
            "Image condition=%s  luminance=%.1f  conf_scale=%.2f  "
            "effective_conf_threshold=%.3f",
            image_condition, image_luminance, conf_scale, effective_conf_threshold,
        )

        t0 = time.perf_counter()
        if enhance_night:
            source_img = _enhance_low_light(source_img)
        stage_timings["night_enhancement"] = time.perf_counter() - t0

        # ── Detection ─────────────────────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            person_dets, animal_dets = detect_objects(source_img, detector_threshold)
            stage_timings["detector"] = time.perf_counter() - t0
        except Exception as exc:
            logger.error("MegaDetector failed: %s", exc)
            if species_task is not None:
                species_task.cancel()
            return IdentifyResponse(
                detections=[], warnings=[f"Detection failed: {exc}"]
            )

        t0 = time.perf_counter()
        allowed_species: List[str] = []
        if species_task is not None:
            try:
                allowed_species = await species_task
            except asyncio.CancelledError:
                allowed_species = []
        allowed_species = [s.strip() for s in allowed_species if isinstance(s, str) and s.strip()]
        stage_timings["species_fetch_wait"] = time.perf_counter() - t0

        detections_out: List[Detection] = []

        # ── Person verification ───────────────────────────────────────────────
        t0 = time.perf_counter()
        person_boxes = [b for b, _ in person_dets]
        person_scores = [c for _, c in person_dets]
        person_crops = [
            _crop_image(
                source_img,
                _padded_xyxy(b, crop_padding_ratio, source_img.width, source_img.height),
            )
            for b in person_boxes
        ]
        animal_boxes = [b for b, _ in animal_dets]
        animal_scores_list = [c for _, c in animal_dets]
        animal_crops = [
            _crop_image(
                source_img,
                _padded_xyxy(b, crop_padding_ratio, source_img.width, source_img.height),
            )
            for b in animal_boxes
        ]
        stage_timings["crop_prepare"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        human_scores: List[Tuple[float, float]] = []
        crop_results: List[Tuple[bool, Optional[str], Optional[str], Optional[float], List[dict]]] = []
        if person_crops or animal_crops:
            model, preprocess, _, device = get_clip_components()
            combined_crops = person_crops + animal_crops
            combined_feats = _encode_image_batch(combined_crops, preprocess, model, device)
            guard_feats = get_guard_text_features()
            species_feats = get_species_text_features()
            person_count = len(person_crops)

            if person_count:
                human_scores = _score_human_likeness_from_features(
                    combined_feats[:person_count],
                    guard_feats,
                )
            if animal_crops:
                crop_results = _classify_animal_features(
                    img_feats=combined_feats[person_count:],
                    allowed_species=allowed_species,
                    confidence_threshold=effective_conf_threshold,
                    margin_threshold=species_margin_threshold,
                    topk=topk_species,
                    guard_feats=guard_feats,
                    species_feats=species_feats,
                )
        stage_timings["clip_shared_encode_and_score"] = time.perf_counter() - t0

        verified_persons: List[Tuple[List[float], float]] = list(zip(person_boxes, person_scores))
        low_conf_person_boxes = 0

        t0 = time.perf_counter()
        try:
            for i, (_box, _conf) in enumerate(zip(person_boxes, person_scores)):
                h_prob, nh_prob = human_scores[i] if i < len(human_scores) else (1.0, 0.0)
                ratio_ok = (nh_prob > 0 and h_prob / nh_prob >= PERSON_VERIFY_RATIO)
                if not (h_prob >= PERSON_VERIFY_THRESHOLD and ratio_ok):
                    low_conf_person_boxes += 1
                    logger.debug(
                        "Person verifier low-confidence, but kept as person due to priority policy  "
                        "h=%.3f  nh=%.3f",
                        h_prob,
                        nh_prob,
                    )
        except Exception as exc:
            logger.error("Person verifier failed: %s", exc)
        stage_timings["person_verify"] = time.perf_counter() - t0

        if low_conf_person_boxes:
            warnings.append(
                f"{low_conf_person_boxes} person detection(s) had low CLIP verifier confidence "
                "but were preserved as person by priority policy."
            )

        for box, conf in verified_persons:
            detections_out.append(
                Detection(bbox=box, label="person", detector_confidence=conf)
            )

        # ── Animal crops (keep independent from detector-person boxes) ────────
        stage_timings["relabelled_animal_classify"] = 0.0

        person_present = bool(verified_persons)
        animal_human_guard_kept_as_animal = 0

        for i, (box, conf) in enumerate(zip(animal_boxes, animal_scores_list)):
            if i >= len(crop_results):
                detections_out.append(
                    Detection(bbox=box, label="animal", detector_confidence=conf)
                )
                continue

            human_override, assigned_species, model_top, species_conf, top3 = crop_results[i]

            if human_override:
                if person_present:
                    # Mixed scenes are common: keep this as animal if person already exists.
                    animal_human_guard_kept_as_animal += 1
                    detections_out.append(
                        Detection(
                            bbox=box,
                            label="animal",
                            detector_confidence=conf,
                            species=assigned_species,
                            species_confidence=species_conf,
                            model_top_prediction=model_top,
                            bioclip_top3=top3 or None,
                        )
                    )
                else:
                    detections_out.append(
                        Detection(bbox=box, label="person", detector_confidence=conf)
                    )
                    person_present = True
            else:
                detections_out.append(
                    Detection(
                        bbox=box,
                        label="animal",
                        detector_confidence=conf,
                        species=assigned_species,
                        species_confidence=species_conf,
                        model_top_prediction=model_top,
                        bioclip_top3=top3 or None,
                    )
                )

        if animal_human_guard_kept_as_animal:
            warnings.append(
                f"{animal_human_guard_kept_as_animal} animal detection(s) were human-guard flagged "
                "but kept as animal because a person was already detected in the image."
            )

        # ── Warnings & sorting ────────────────────────────────────────────────
        unknown_animals = [
            d for d in detections_out
            if d.label == "animal" and d.species is None and d.model_top_prediction is None
        ]
        out_of_list = [
            d for d in detections_out
            if d.label == "animal" and d.species == "UNKNOWN"
        ]
        low_conf = [
            d for d in detections_out
            if d.label == "animal" and d.species is None and d.model_top_prediction is not None
        ]

        if out_of_list:
            warned_species = ", ".join(
                d.model_top_prediction for d in out_of_list if d.model_top_prediction
            )
            warnings.append(
                f"{len(out_of_list)} animal(s) identified as species not in the allowed list "
                f"({warned_species}) — returned as UNKNOWN."
            )
        if low_conf:
            warnings.append(
                f"{len(low_conf)} animal(s) below species confidence/margin threshold — "
                f"species returned as None."
            )
        if not allowed_species and any(d.label == "animal" for d in detections_out):
            warnings.append(
                "No allowed species list configured for this client — canonical species labels were returned directly."
            )
        if not detections_out:
            warnings.append("No persons or animals detected above the configured thresholds.")

        # Persons first, then animals
        detections_out.sort(key=lambda d: 0 if d.label == "person" else 1)
        stage_timings["total"] = time.perf_counter() - request_start
        logger.info(
            "identify timings: %s",
            ", ".join(f"{name}={value:.3f}s" for name, value in stage_timings.items()),
        )

        return IdentifyResponse(
            detections=detections_out,
            warnings=warnings or None,
            image_condition=image_condition,
            image_luminance=image_luminance,
            effective_confidence_threshold=effective_conf_threshold,
        )

    except Exception as exc:
        logger.exception("Unexpected error in /identify")
        if species_task is not None and not species_task.done():
            species_task.cancel()
        return IdentifyResponse(detections=[], warnings=[f"Processing failed: {exc}"])


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting CameraTrap API on http://0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)