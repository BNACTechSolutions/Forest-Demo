import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

import httpx
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from speciesnet import SpeciesNet


app = FastAPI(title="SpeciesNet Adapter API")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("speciesnet_adapter")

cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
SPECIESNET_TIMEOUT_SECONDS = int(os.getenv("SPECIESNET_TIMEOUT_SECONDS", "900"))
SPECIESNET_MODEL = os.getenv("SPECIESNET_MODEL", "kaggle:google/speciesnet/pyTorch/v4.0.2a/1")
SPECIESNET_RUN_MODE = os.getenv("SPECIESNET_RUN_MODE", "single_thread")
SPECIESNET_GEOFENCE = os.getenv("SPECIESNET_GEOFENCE", "false").lower() == "true"
SPECIESNET_PRELOAD = os.getenv("SPECIESNET_PRELOAD", "true").lower() == "true"

DEFAULT_SPECIES = ["Dog", "Cat", "Cow"]

_speciesnet_model: Optional[SpeciesNet] = None
_speciesnet_model_lock = threading.Lock()


class Detection(BaseModel):
    bbox: List[float]
    label: str
    detector_confidence: float
    species: Optional[str] = None
    species_confidence: Optional[float] = None
    mask_png_base64: Optional[str] = None
    bioclip_top3: Optional[List[dict]] = None


class IdentifyResponse(BaseModel):
    detections: List[Detection]
    warnings: Optional[List[str]] = None


async def fetch_client_species(client_id: str) -> List[str]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BACKEND_URL}/api/species-of-interest/supported")
            if response.status_code == 200:
                data = response.json()
                species_list = [s.get("specieName") for s in data.get("supportedSpecies", [])]
                species_list = [s for s in species_list if isinstance(s, str) and s.strip()]
                if species_list:
                    logger.info("Loaded %d client species for %s", len(species_list), client_id)
                    return species_list
    except Exception as e:
        logger.warning("Failed to fetch species list from backend: %s", e)

    return DEFAULT_SPECIES


def normalize_label(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()


def get_speciesnet_model() -> SpeciesNet:
    global _speciesnet_model
    if _speciesnet_model is not None:
        return _speciesnet_model

    with _speciesnet_model_lock:
        if _speciesnet_model is None:
            logger.info("Loading SpeciesNet model (components=all): %s", SPECIESNET_MODEL)
            start = time.time()
            _speciesnet_model = SpeciesNet(
                SPECIESNET_MODEL,
                components="all",
                geofence=SPECIESNET_GEOFENCE,
            )
            logger.info("SpeciesNet model loaded in %.2fs", time.time() - start)
    return _speciesnet_model


def run_speciesnet_on_image(image_bytes: bytes, suffix: str = ".jpg") -> dict:
    with tempfile.TemporaryDirectory(prefix="speciesnet-run-") as tmp_dir:
        image_path = Path(tmp_dir) / f"image{suffix if suffix else '.jpg'}"
        image_path.write_bytes(image_bytes)

        with Image.open(image_path) as img:
            img.verify()

        model = get_speciesnet_model()
        start = time.time()
        result = model.predict(
            instances_dict={"instances": [{"filepath": str(image_path)}]},
            run_mode=SPECIESNET_RUN_MODE,
            progress_bars=False,
        )
        elapsed = time.time() - start
        logger.info("SpeciesNet inference completed in %.2fs", elapsed)

        if elapsed > SPECIESNET_TIMEOUT_SECONDS:
            raise RuntimeError(
                f"SpeciesNet exceeded timeout budget ({elapsed:.2f}s > {SPECIESNET_TIMEOUT_SECONDS}s)"
            )

        predictions = (result or {}).get("predictions") or []
        if not predictions:
            raise ValueError("SpeciesNet returned no predictions")

        return predictions[0]


@app.on_event("startup")
def preload_speciesnet_model() -> None:
    if not SPECIESNET_PRELOAD:
        logger.info("SpeciesNet preload disabled via SPECIESNET_PRELOAD=false")
        return
    try:
        get_speciesnet_model()
    except Exception as e:
        logger.error("SpeciesNet preload failed: %s", e)


def map_classifier_to_allowed_species(classifications: dict, allowed_species: List[str], topk: int = 3):
    classes = classifications.get("classes") or []
    scores = classifications.get("scores") or []
    pairs = [(str(c), float(s)) for c, s in zip(classes, scores)]

    blank_like_terms = {"blank", "empty", "background", "unknown"}
    if pairs:
        top_label_norm = normalize_label(pairs[0][0])
        if any(term in top_label_norm for term in blank_like_terms):
            return None, None, []

    allowed_norm = {normalize_label(s): s for s in allowed_species}
    candidate_matches = []
    for label, score in pairs:
        norm = normalize_label(label)
        for allow_norm, allow_raw in allowed_norm.items():
            if allow_norm and allow_norm in norm:
                candidate_matches.append((allow_raw, score, label))

    if not candidate_matches:
        return None, None, []

    candidate_matches.sort(key=lambda x: x[1], reverse=True)
    top_species = candidate_matches[0][0]
    top_conf = float(candidate_matches[0][1])

    topk_formatted = []
    used = set()
    for species_name, score, source_label in candidate_matches:
        key = normalize_label(species_name)
        if key in used:
            continue
        used.add(key)
        topk_formatted.append(
            {
                "species": species_name,
                "confidence": round(float(score), 4),
                "source_label": source_label,
            }
        )
        if len(topk_formatted) >= topk:
            break

    return top_species, top_conf, topk_formatted


@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    file: UploadFile = File(...),
    client_id: Optional[str] = Form(None),
    detector_threshold: float = Form(0.45),
    poacher_threshold: float = Form(0.12),
    species_confidence_threshold: float = Form(0.45),
    topk_species: int = Form(5),
):
    warnings: List[str] = []

    if not file.filename:
        return IdentifyResponse(detections=[], warnings=["No file provided."])

    if file.content_type not in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
        return IdentifyResponse(detections=[], warnings=["Unsupported file type. Use jpg/png/webp."])

    detector_threshold = min(max(float(detector_threshold), 0.05), 1.0)
    poacher_threshold = min(max(float(poacher_threshold), 0.05), 1.0)
    species_confidence_threshold = min(max(float(species_confidence_threshold), 0.05), 1.0)
    topk_species = int(min(max(int(topk_species), 3), 10))

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return IdentifyResponse(detections=[], warnings=[f"File too large. Max {MAX_UPLOAD_MB} MB."])

    allowed_species = DEFAULT_SPECIES
    if client_id:
        allowed_species = await fetch_client_species(client_id)

    suffix = Path(file.filename).suffix.lower() or ".jpg"

    try:
        pred = run_speciesnet_on_image(image_bytes, suffix=suffix)
    except Exception as e:
        logger.error("SpeciesNet execution failed: %s", e)
        return IdentifyResponse(detections=[], warnings=[f"SpeciesNet failed: {e}"])

    dets = pred.get("detections") or []
    classifications = pred.get("classifications") or {}

    top_species, top_species_conf, top3 = map_classifier_to_allowed_species(
        classifications,
        allowed_species,
        topk=min(3, topk_species),
    )

    output: List[Detection] = []

    for d in dets:
        conf = float(d.get("conf", 0.0))
        category = str(d.get("category", ""))
        label_raw = str(d.get("label", "")).lower().strip()
        bbox = d.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue

        out_bbox = [float(v) for v in bbox]

        if category == "2" or label_raw == "human":
            if conf >= poacher_threshold:
                output.append(
                    Detection(
                        bbox=out_bbox,
                        label="person",
                        detector_confidence=conf,
                    )
                )
            continue

        if category == "1" or label_raw == "animal":
            if conf < detector_threshold:
                continue

            if top_species is None or top_species_conf is None:
                output.append(
                    Detection(
                        bbox=out_bbox,
                        label="animal",
                        detector_confidence=conf,
                        species=None,
                        species_confidence=None,
                        bioclip_top3=top3 if top3 else None,
                    )
                )
                warnings.append("Animal detected but species withheld due to low-confidence species mapping.")
                continue

            if top_species_conf < species_confidence_threshold:
                output.append(
                    Detection(
                        bbox=out_bbox,
                        label="animal",
                        detector_confidence=conf,
                        species=None,
                        species_confidence=None,
                        bioclip_top3=top3 if top3 else None,
                    )
                )
                warnings.append(
                    f"Animal detected but species confidence below threshold ({top_species_conf:.3f} < {species_confidence_threshold:.2f})."
                )
                continue

            output.append(
                Detection(
                    bbox=out_bbox,
                    label="animal",
                    detector_confidence=conf,
                    species=top_species,
                    species_confidence=float(top_species_conf),
                    bioclip_top3=top3 if top3 else None,
                )
            )

    if not output and not warnings:
        warnings.append("No persons or animals detected above configured thresholds.")

    output.sort(key=lambda x: 0 if x.label == "person" else 1)

    return IdentifyResponse(detections=output, warnings=warnings or None)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting SpeciesNet Adapter on http://0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
