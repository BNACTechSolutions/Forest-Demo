# bioclip_server_improved.py
import io
import base64
import logging
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
    from megadetector.detection.run_detector import load_detector
    MEGADETECTOR_AVAILABLE = True
except Exception as e:
    print(f"MegaDetector package not available: {e}")
    MEGADETECTOR_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False

app = FastAPI(title="BioCLIP Multi-Object Identification API")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bioclip_server")

cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Config / Backend URL
# -------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "120000000"))
logger.info("Backend URL configured: %s", BACKEND_URL)

# Default fallback list
DEFAULT_SPECIES = ["Dog", "Cat", "Cow"]

candidate_species = DEFAULT_SPECIES

# -------------------------
# Load BioCLIP model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Device: %s", device)

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
model = model.to(device)
model.eval()

# -------------------------
# MegaDetector v5 for poacher/animal detection
# -------------------------
DETECTOR = None
if MEGADETECTOR_AVAILABLE:
    try:
        DETECTOR = load_detector("models/md_v5a.0.0.pt")
        logger.info("MegaDetector v5 loaded successfully via megadetector package.")
        logger.info("MegaDetector classes: 1=Animal, 2=Person")
    except Exception as e:
        logger.warning("Failed to load MegaDetector model from local path: %s", e)
        logger.info("Attempting fallback to MDV5A identifier...")
        try:
            # Fallback to MDv5a default
            DETECTOR = load_detector("MDV5A")
            logger.info("MegaDetector v5 loaded via default MDV5A identifier.")
        except Exception as e2:
            logger.error("MegaDetector fallback failed: %s", e2)
            DETECTOR = None
else:
    logger.error("megadetector package not available. Install with: pip install megadetector")

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
        logger.info("SAM loaded.")
    except Exception as e:
        logger.warning("SAM not loaded: %s", e)
else:
    logger.info("segment-anything not available.")

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
                    logger.info("Loaded %d species from backend for client %s", len(species_list), client_id)
                    return species_list
    except Exception as e:
        logger.warning("Error fetching species from backend: %s", e)
    
    logger.info("Falling back to default species list")
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

def classify_with_bioclip(pil_crop: Image.Image, allowed_species: List[str], confidence_threshold: float = 0.3, topk: int = 3) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
    """
    Classify crop with BioCLIP using only the allowed species list.
    Returns (species, confidence, top_k_predictions) with highest confidence, or None if below threshold.
    """
    if not allowed_species:
        return None
    
    species_list = list(set(allowed_species))  # Remove duplicates if any
    text_tokens = tokenizer(species_list).to(device)
    
    img_t = preprocess_val(pil_crop).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(img_t)
        text_features = model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # Get top-k predictions
    topk_vals, topk_idxs = logits[0].topk(min(topk, len(species_list)))
    
    max_conf = float(topk_vals[0])
    max_idx = int(topk_idxs[0])
    
    if max_conf < confidence_threshold:
        return None
    
    predicted_species = species_list[max_idx]
    
    # Get all top-k for debugging
    all_topk = [(species_list[int(idx)], float(val)) for idx, val in zip(topk_idxs, topk_vals)]
    
    return predicted_species, max_conf, all_topk

def preprocess_for_ir_night(pil_img: Image.Image) -> Image.Image:
    """Preprocess IR/night images for better detection."""
    from PIL import ImageOps
    
    # Convert to grayscale for consistent processing
    img_gray = ImageOps.grayscale(pil_img)
    
    # Apply autocontrast to enhance low-contrast night/IR images
    img_enhanced = ImageOps.autocontrast(img_gray)
    
    # Convert back to RGB for detector compatibility
    img_rgb = img_enhanced.convert("RGB")
    
    return img_rgb

def _xywh_iou(box_a: List[float], box_b: List[float]) -> float:
    """IoU for normalized MegaDetector boxes [x, y, w, h]."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom

def dedupe_md_detections(detections: List[dict], iou_threshold: float = 0.6) -> List[dict]:
    """Deduplicate MegaDetector detections, keeping highest confidence overlaps per category."""
    ordered = sorted(detections, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
    kept: List[dict] = []

    for det in ordered:
        det_cat = str(det.get("category", ""))
        det_bbox = det.get("bbox", None)
        if not det_bbox or len(det_bbox) != 4:
            continue

        duplicate = False
        for kept_det in kept:
            kept_cat = str(kept_det.get("category", ""))
            if det_cat != kept_cat:
                continue
            kept_bbox = kept_det.get("bbox", None)
            if not kept_bbox or len(kept_bbox) != 4:
                continue
            if _xywh_iou(det_bbox, kept_bbox) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    return kept

def is_low_information_crop(pil_crop: Image.Image, std_threshold: float = 10.0, dark_threshold: float = 18.0) -> bool:
    """Reject mostly-black/flat crops that cause false species predictions."""
    gray = pil_crop.convert("L")
    arr = np.array(gray, dtype=np.float32)
    if arr.size == 0:
        return True
    return float(arr.std()) < std_threshold or float(arr.mean()) < dark_threshold

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
    poacher_threshold: float = Form(0.15),
    species_confidence_threshold: float = Form(0.7),
    species_margin_threshold: float = Form(0.15),
    low_info_std_threshold: float = Form(10.0),
    topk_species: int = Form(5),
):
    """
    Mission-critical poacher and animal detection system for forest environments.
    
    Parameters:
    - file: Image file (IR/Night vision supported)
    - client_id: Client ID to fetch their supported species
    - run_sam: Whether to run SAM mask generator
    - detector_threshold: Minimum MegaDetector confidence for animals (default 0.35)
    - poacher_threshold: Minimum confidence for person detection (default 0.15 - high recall)
    - species_confidence_threshold: Minimum BioCLIP confidence for animal species (default 0.7)
    - species_margin_threshold: Top1-Top2 confidence gap for species acceptance (default 0.15)
    - low_info_std_threshold: Reject low-detail crops before species classification (default 10.0)
    
    MegaDetector classes: 1=Animal, 2=Person
    """
    # Fetch client's allowed species list
    allowed_species_list = candidate_species
    if client_id:
        allowed_species_list = await fetch_client_species(client_id)

    # Basic production input validation
    if not file.filename:
        return IdentifyResponse(detections=[], warnings=["No file provided."])
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
        return IdentifyResponse(detections=[], warnings=["Unsupported file type. Use jpg/png/webp."])

    detector_threshold = min(max(float(detector_threshold), 0.05), 1.0)
    poacher_threshold = min(max(float(poacher_threshold), 0.05), 1.0)
    species_confidence_threshold = min(max(float(species_confidence_threshold), 0.5), 1.0)
    species_margin_threshold = min(max(float(species_margin_threshold), 0.05), 0.5)
    low_info_std_threshold = min(max(float(low_info_std_threshold), 2.0), 30.0)
    topk_species = int(min(max(int(topk_species), 3), 10))
    
    img_bytes = await file.read()
    if len(img_bytes) > MAX_UPLOAD_BYTES:
        return IdentifyResponse(detections=[], warnings=[f"File too large. Max {MAX_UPLOAD_MB} MB."])

    pil_img = pil_from_bytes(img_bytes)
    img_w, img_h = pil_img.size

    warnings = []
    detections_out: List[Detection] = []

    # If no detector available, return empty
    if DETECTOR is None:
        warnings.append("MegaDetector not available. Cannot process image.")
        return IdentifyResponse(detections=[], warnings=warnings)

    # Preprocess for IR/Night images: enhance contrast and normalize
    pil_img_processed = preprocess_for_ir_night(pil_img)
    
    # Run MegaDetector v5 in dual-pass mode (original + enhanced) for better person recall
    try:
        img_array_original = np.array(pil_img)
        img_array_processed = np.array(pil_img_processed)

        result_original = DETECTOR.generate_detections_one_image(img_array_original)
        result_processed = DETECTOR.generate_detections_one_image(img_array_processed)

        detections_original = result_original.get('detections', [])
        detections_processed = result_processed.get('detections', [])
        detections_list = dedupe_md_detections(detections_original + detections_processed)
        
        if not detections_list:
            warnings.append("No persons or animals detected in image.")
            return IdentifyResponse(detections=[], warnings=warnings)
            
    except Exception as e:
        warnings.append(f"MegaDetector inference failed: {e}")
        return IdentifyResponse(detections=[], warnings=warnings)

    # MegaDetector class mapping: 0=Animal, 1=Person
    MD_CLASS_ANIMAL = 1  # MegaDetector uses 1 for animal
    MD_CLASS_PERSON = 2  # MegaDetector uses 2 for person

    # Convert MegaDetector format to our format
    valid_boxes = []
    for det in detections_list:
        try:
            category = int(det['category'])
            conf = float(det['conf'])
            bbox = det['bbox']  # [x, y, w, h] in normalized coordinates [0-1]
            
            # Convert normalized [x, y, w, h] to pixel [x1, y1, x2, y2]
            x, y, w, h = bbox
            x1 = x * img_w
            y1 = y * img_h
            x2 = x1 + (w * img_w)
            y2 = y1 + (h * img_h)
            box = [x1, y1, x2, y2]
            
            # POACHER-FIRST LOGIC: Use lower threshold for person detection
            if category == MD_CLASS_PERSON:
                if conf >= poacher_threshold:
                    valid_boxes.append((box, conf, "person", category))
            # Animals use standard threshold
            elif category == MD_CLASS_ANIMAL:
                if conf >= detector_threshold:
                    valid_boxes.append((box, conf, "animal", category))
                    
        except Exception as e:
            logger.warning("Error parsing MegaDetector detection: %s", e)
            continue

    # Check if we have valid detections after filtering
    if not valid_boxes:
        warnings.append("No persons or animals detected above confidence thresholds.")
        return IdentifyResponse(detections=[], warnings=warnings)

    # Count detection types for comprehensive reporting
    person_count = sum(1 for _, _, label, _ in valid_boxes if label == "person")
    animal_count = sum(1 for _, _, label, _ in valid_boxes if label == "animal")
    
    logger.info("MegaDetector results: total=%d persons=%d animals=%d", len(valid_boxes), person_count, animal_count)
    
    # Optionally compute SAM masks
    sam_masks_by_box = {}
    if run_sam and SAM_MASK_GENERATOR is not None:
        arr = np.array(pil_img)
        all_masks = SAM_MASK_GENERATOR.generate(arr)
        
        for i, (box, conf, label, md_class_id) in enumerate(valid_boxes):
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

    # Separate detections by type for organized processing
    poacher_detections = []
    animal_detections = []
    
    # Process each detection with POACHER-FIRST logic
    for i, (box, det_conf, label, md_class_id) in enumerate(valid_boxes):
        x1, y1, x2, y2 = box
        
        # CRITICAL: POACHER DETECTION - No classification needed, immediate alert
        if label == "person":
            det = Detection(
                bbox=[x1, y1, x2, y2],
                label="person",
                detector_confidence=det_conf,
                species=None,
                species_confidence=None,
                mask_png_base64=mask_to_base64_png(sam_masks_by_box[i]) if i in sam_masks_by_box else None
            )
            poacher_detections.append(det)
            logger.warning("CRITICAL person detected idx=%d conf=%.4f", i + 1, det_conf)
            continue
        

        # For animals ONLY - run BioCLIP species classification
        # NOTE: ALL ANIMALS ARE PROCESSED REGARDLESS OF POACHER PRESENCE
        if label == "animal":
            crop = crop_pil(pil_img, box)  # Crop from ORIGINAL color image, not processed

            if is_low_information_crop(crop, std_threshold=low_info_std_threshold):
                warnings.append("Low-information animal crop rejected (likely empty/dark frame noise).")
                continue
            
            try:
                result = classify_with_bioclip(
                    crop, 
                    allowed_species=allowed_species_list, 
                    confidence_threshold=species_confidence_threshold, 
                    topk=max(3, topk_species)
                )
                
                if result is not None:
                    predicted_species, species_conf, all_topk = result
                    top2_conf = all_topk[1][1] if len(all_topk) > 1 else 0.0
                    conf_margin = species_conf - float(top2_conf)

                    if conf_margin < species_margin_threshold:
                        warnings.append(
                            f"Animal species rejected due to low confidence margin "
                            f"({conf_margin:.3f} < {species_margin_threshold})."
                        )
                        continue
                    
                    # Format top-k for response
                    topk_formatted = [{"species": s, "confidence": round(c, 4)} for s, c in all_topk[:3]]
                    
                    det = Detection(
                        bbox=[x1, y1, x2, y2],
                        label="animal",
                        detector_confidence=det_conf,
                        species=predicted_species,
                        species_confidence=species_conf,
                        mask_png_base64=mask_to_base64_png(sam_masks_by_box[i]) if i in sam_masks_by_box else None,
                        bioclip_top3=topk_formatted
                    )
                    animal_detections.append(det)
                    logger.info("Animal detected idx=%d species=%s conf=%.4f", i + 1, predicted_species, species_conf)
                else:
                    # Below confidence threshold - reject instead of forcing a random species
                    warnings.append(
                        f"Animal detected by MegaDetector but BioCLIP species confidence too low "
                        f"(< {species_confidence_threshold}). Detection dropped."
                    )
            
            except Exception as e:
                warnings.append(f"BioCLIP classification failed for animal detection {i}: {e}")
                logger.warning("Animal detection classification error idx=%d err=%s", i + 1, e)
                continue

    # Combine detections in priority order: POACHERS FIRST, then animals
    detections_out = poacher_detections + animal_detections
    
    # Log final summary
    logger.info("Processing complete: persons=%d animals=%d", len(poacher_detections), len(animal_detections))
    
    return IdentifyResponse(detections=detections_out, warnings=warnings)


# -------------------------
# Server startup
# -------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting BioCLIP Poacher Detection Server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)