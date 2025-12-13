# bioclip_server_improved.py
import io
import base64
import json
from typing import List, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
import open_clip
import httpx
import os

# Optional model imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False

app = FastAPI(title="BioCLIP Multi-Object Identification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Config / Backend URL
# -------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")
print(f"Backend URL: {BACKEND_URL}")

# Default fallback list
DEFAULT_SPECIES = ["Dog", "Cat", "Cow"]

# EXPANDED species list for BioCLIP classification
# This ensures BioCLIP can distinguish between similar animals
BIOCLIP_EXTENDED_SPECIES = [
    # Canines
    "Dog", "Wolf", "Fox", "Coyote", "Jackal", "Dingo",
    # Felines
    "Cat", "Lion", "Tiger", "Leopard", "Cheetah", "Jaguar", "Panther", "Lynx", "Puma", "Cougar",
    # Bovines
    "Cow", "Bull", "Buffalo", "Bison", "Ox", "Yak", "Water Buffalo",
    # Equines
    "Horse", "Zebra", "Donkey", "Mule",
    # Primates (NOT humans)
    "Monkey", "Chimpanzee", "Gorilla", "Orangutan", "Baboon", "Lemur", "Gibbon", "Macaque",
    # Bears
    "Bear", "Grizzly Bear", "Polar Bear", "Black Bear", "Panda",
    # Birds
    "Bird", "Eagle", "Hawk", "Owl", "Parrot", "Crow", "Pigeon", "Sparrow", "Penguin", "Flamingo",
    # Reptiles
    "Snake", "Lizard", "Crocodile", "Alligator", "Turtle", "Tortoise", "Iguana",
    # Rodents
    "Mouse", "Rat", "Squirrel", "Hamster", "Guinea Pig", "Rabbit", "Beaver",
    # Marine
    "Dolphin", "Whale", "Seal", "Sea Lion", "Otter",
    # Farm/Domestic
    "Sheep", "Goat", "Pig", "Chicken", "Duck", "Goose", "Turkey",
    # Wild
    "Elephant", "Giraffe", "Rhinoceros", "Hippopotamus", "Kangaroo", "Deer", "Moose", "Elk",
]

candidate_species = DEFAULT_SPECIES

# -------------------------
# Load BioCLIP model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
model = model.to(device)
model.eval()

# Tokenize the EXTENDED species list for BioCLIP
bioclip_text_tokens = tokenizer(BIOCLIP_EXTENDED_SPECIES).to(device)

# -------------------------
# Optional: YOLO object detector
# -------------------------
DETECTOR = None
if YOLO_AVAILABLE:
    try:
        DETECTOR = YOLO("yolov8n.pt")
        print("YOLO detector loaded.")
    except Exception as e:
        print("Failed to load YOLO model:", e)
        DETECTOR = None
else:
    print("ultralytics YOLO not available. Install with: pip install ultralytics")

# -------------------------
# Optional: SAM mask generator
# -------------------------
SAM = None
SAM_MASK_GENERATOR = None
if SAM_AVAILABLE:
    try:
        sam_checkpoint = "models/sam_vit_b_01ec64.pth"
        sam_model_type = "vit_b"
        SAM = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(device)
        SAM_MASK_GENERATOR = SamAutomaticMaskGenerator(SAM)
        print("SAM loaded.")
    except Exception as e:
        print("SAM not loaded.", e)
else:
    print("segment-anything not available.")

# -------------------------
# Helpers
# -------------------------
async def fetch_client_species(client_id: str) -> List[str]:
    """Fetch supported species for a client from the backend."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{BACKEND_URL}/api/species-of-interest/supported"
            )
            if response.status_code == 200:
                data = response.json()
                species_list = [s.get("specieName") for s in data.get("supportedSpecies", [])]
                if species_list:
                    print(f"Loaded {len(species_list)} species from backend for client {client_id}")
                    return species_list
    except Exception as e:
        print(f"Error fetching species from backend: {e}")
    
    print(f"Falling back to default species list")
    return DEFAULT_SPECIES

def pil_from_bytes(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return img

def crop_pil(img: Image.Image, box: List[float]) -> Image.Image:
    x1, y1, x2, y2 = [int(round(x)) for x in box]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.width)
    y2 = min(y2, img.height)
    return img.crop((x1, y1, x2, y2))

def mask_to_base64_png(mask: np.ndarray) -> str:
    """Convert mask to base64-encoded PNG."""
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def classify_with_bioclip(pil_crop: Image.Image, confidence_threshold: float = 0.5, topk: int = 3) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
    """
    Classify crop with BioCLIP using EXTENDED species list.
    Returns (species, confidence, top_k_predictions) with highest confidence, or None if below threshold.
    """
    img_t = preprocess_val(pil_crop).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(img_t)
        text_features = model.encode_text(bioclip_text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # Get top-k predictions
    topk_vals, topk_idxs = logits[0].topk(min(topk, len(BIOCLIP_EXTENDED_SPECIES)))
    
    max_conf = float(topk_vals[0])
    max_idx = int(topk_idxs[0])
    
    if max_conf < confidence_threshold:
        return None
    
    predicted_species = BIOCLIP_EXTENDED_SPECIES[max_idx]
    
    # Get all top-k for debugging
    all_topk = [(BIOCLIP_EXTENDED_SPECIES[int(idx)], float(val)) for idx, val in zip(topk_idxs, topk_vals)]
    
    return predicted_species, max_conf, all_topk

def is_allowed_species(predicted_species: str, allowed_list: List[str]) -> bool:
    """
    Check if predicted species is in the allowed list (case-insensitive exact match).
    """
    predicted_lower = predicted_species.lower()
    for allowed in allowed_list:
        if allowed.lower() == predicted_lower:
            return True
    return False

def is_animal_class(class_name: str) -> bool:
    """Check if YOLO class name represents an animal we care about."""
    class_lower = class_name.lower()
    
    # Person is not an animal
    if "person" in class_lower or "human" in class_lower:
        return False
    
    # Common COCO animal classes
    animal_keywords = [
        "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", 
        "zebra", "giraffe", "deer", "bird", "snake", "lizard"
    ]
    
    return any(keyword in class_lower for keyword in animal_keywords)

# -------------------------
# Response models
# -------------------------
class Detection(BaseModel):
    bbox: List[float]
    label: str
    detector_confidence: float
    species: Optional[str] = None
    species_confidence: Optional[float] = None
    mask_png_base64: Optional[str] = None
    bioclip_top3: Optional[List[dict]] = None  # Top 3 BioCLIP predictions for debugging

class IdentifyResponse(BaseModel):
    detections: List[Detection]
    warnings: Optional[List[str]] = None

# -------------------------
# Main endpoint
# -------------------------
@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    file: UploadFile = File(...),
    client_id: str = Form(None),
    run_sam: bool = Form(False),
    detector_threshold: float = Form(0.35),
    species_confidence_threshold: float = Form(0.3),  # LOWERED from 0.5 to 0.3
    topk_species: int = Form(5),
):
    """
    Upload an image for identification.
    
    Parameters:
    - file: Image file
    - client_id: Client ID to fetch their supported species
    - run_sam: Whether to run SAM mask generator
    - detector_threshold: Minimum YOLO confidence (0-1, default 0.35)
    - species_confidence_threshold: Minimum BioCLIP confidence (0-1, default 0.3)
    """
    # Fetch client's allowed species list
    allowed_species_list = candidate_species
    if client_id:
        allowed_species_list = await fetch_client_species(client_id)
    
    img_bytes = await file.read()
    pil_img = pil_from_bytes(img_bytes)
    img_w, img_h = pil_img.size

    warnings = []
    detections_out: List[Detection] = []

    # If no detector available, return empty
    if DETECTOR is None:
        warnings.append("Object detector not available. Cannot process image.")
        return IdentifyResponse(detections=[], warnings=warnings)

    # Run YOLO detector
    results = DETECTOR(pil_img)
    results0 = results[0]

    # Parse YOLO results
    try:
        xyxy = results0.boxes.xyxy.cpu().numpy()
        confs = results0.boxes.conf.cpu().numpy()
        cls_ids = results0.boxes.cls.cpu().numpy().astype(int)
    except Exception:
        try:
            data = results0.boxes.data.cpu().numpy()
            xyxy = data[:, :4]
            confs = data[:, 4]
            cls_ids = data[:, 5].astype(int)
        except Exception as e:
            warnings.append(f"Unable to parse detector output: {e}")
            return IdentifyResponse(detections=[], warnings=warnings)

    names_map = getattr(results0, "names", None)

    # Filter valid detections (persons and animals only)
    valid_boxes = []
    for box, conf, cls_id in zip(xyxy, confs, cls_ids):
        if conf < detector_threshold:
            continue
        
        # Get class name
        if names_map is not None:
            class_name = names_map.get(int(cls_id), str(cls_id))
        else:
            class_name = f"class_{int(cls_id)}"
        
        class_lower = class_name.lower()
        
        # Determine label
        if "person" in class_lower or "human" in class_lower:
            label = "person"
            valid_boxes.append((box.tolist(), float(conf), label, class_name))
        elif is_animal_class(class_name):
            label = "animal"
            valid_boxes.append((box.tolist(), float(conf), label, class_name))
        else:
            # Not a person or animal - ignore (suitcase, chair, etc.)
            continue

    # If no valid detections, return empty
    if not valid_boxes:
        warnings.append("No persons or animals detected in image.")
        return IdentifyResponse(detections=[], warnings=warnings)

    # Optionally compute SAM masks
    sam_masks_by_box = {}
    if run_sam and SAM_MASK_GENERATOR is not None:
        arr = np.array(pil_img)
        all_masks = SAM_MASK_GENERATOR.generate(arr)
        
        for i, (box, conf, label, class_name) in enumerate(valid_boxes):
            best_mask = None
            for m in all_masks:
                mask_bool = m["segmentation"]
                mbbox = m["bbox"]
                mbx1, mby1, mbw, mbh = mbbox
                mbx2 = mbx1 + mbw
                mby2 = mby1 + mbh
                
                if not (mbx2 < box[0] or mbx1 > box[2] or mby2 < box[1] or mby1 > box[3]):
                    best_mask = mask_bool
                    break
            
            if best_mask is not None:
                sam_masks_by_box[i] = best_mask

    # Process each detection
    for i, (box, det_conf, label, class_name) in enumerate(valid_boxes):
        x1, y1, x2, y2 = box
        
        # For persons, just add the detection
        if label == "person":
            det = Detection(
                bbox=[x1, y1, x2, y2],
                label="person",
                detector_confidence=det_conf,
                species=None,
                species_confidence=None,
                mask_png_base64=mask_to_base64_png(sam_masks_by_box[i]) if i in sam_masks_by_box else None
            )
            detections_out.append(det)
            continue
        
        # For animals, run BioCLIP classification
        crop = crop_pil(pil_img, box)
        
        try:
            result = classify_with_bioclip(crop, confidence_threshold=species_confidence_threshold, topk=5)
            
            if result is not None:
                predicted_species, species_conf, all_topk = result
                
                # Format top-k for response
                topk_formatted = [{"species": s, "confidence": round(c, 4)} for s, c in all_topk[:3]]
                
                # Try to find ANY match in top-k predictions that's in allowed list
                matched_species = None
                matched_conf = None
                
                for pred_species, pred_conf in all_topk:
                    if is_allowed_species(pred_species, allowed_species_list):
                        matched_species = pred_species
                        matched_conf = pred_conf
                        break
                
                if matched_species:
                    # Found a match in top-k
                    det = Detection(
                        bbox=[x1, y1, x2, y2],
                        label="animal",
                        detector_confidence=det_conf,
                        species=matched_species,
                        species_confidence=matched_conf,
                        mask_png_base64=mask_to_base64_png(sam_masks_by_box[i]) if i in sam_masks_by_box else None,
                        bioclip_top3=topk_formatted
                    )
                    detections_out.append(det)
                else:
                    # No match in allowed list
                    warnings.append(
                        f"Animal detected (YOLO: {class_name}). BioCLIP top-3: {topk_formatted}. "
                        f"None match your species list."
                    )
            else:
                # Below confidence threshold
                warnings.append(
                    f"Animal detected (YOLO: {class_name}) but BioCLIP confidence too low "
                    f"(< {species_confidence_threshold})."
                )
        
        except Exception as e:
            warnings.append(f"BioCLIP classification failed for box {i}: {e}")

    return IdentifyResponse(detections=detections_out, warnings=warnings)