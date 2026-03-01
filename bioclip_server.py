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

# -------------------------
# MegaDetector v5 for poacher/animal detection
# -------------------------
DETECTOR = None
if MEGADETECTOR_AVAILABLE:
    try:
        DETECTOR = load_detector("models/md_v5a.0.0.pt")
        print("MegaDetector v5 loaded successfully via megadetector package.")
        print("MegaDetector classes: 1=Animal, 2=Person")
    except Exception as e:
        print("Failed to load MegaDetector model:", e)
        print("Attempting fallback to direct path...")
        try:
            # Fallback to MDv5a default
            DETECTOR = load_detector("MDV5A")
            print("MegaDetector v5 loaded via default MDV5A identifier.")
        except Exception as e2:
            print("Fallback failed:", e2)
            DETECTOR = None
else:
    print("megadetector package not available. Install with: pip install megadetector")

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
    poacher_threshold: float = Form(0.25),  # Lower threshold for critical poacher detection
    species_confidence_threshold: float = Form(0.3),
    topk_species: int = Form(5),
):
    """
    Mission-critical poacher and animal detection system for forest environments.
    
    Parameters:
    - file: Image file (IR/Night vision supported)
    - client_id: Client ID to fetch their supported species
    - run_sam: Whether to run SAM mask generator
    - detector_threshold: Minimum MegaDetector confidence for animals (default 0.35)
    - poacher_threshold: Minimum confidence for person detection (default 0.25 - no poacher missed!)
    - species_confidence_threshold: Minimum BioCLIP confidence for animal species (default 0.3)
    
    MegaDetector classes: 1=Animal, 2=Person
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
        warnings.append("MegaDetector not available. Cannot process image.")
        return IdentifyResponse(detections=[], warnings=warnings)

    # Preprocess for IR/Night images: enhance contrast and normalize
    pil_img_processed = preprocess_for_ir_night(pil_img)
    
    # Run MegaDetector v5 - convert PIL to numpy array
    try:
        # Convert to numpy array for MegaDetector
        img_array = np.array(pil_img_processed)
        
        # MegaDetector API: generate_detections_one_image returns dict with 'detections' list
        result = DETECTOR.generate_detections_one_image(img_array)
        
        # Parse MegaDetector output format
        # Each detection: {'category': '1', 'conf': 0.95, 'bbox': [x, y, w, h]}
        detections_list = result.get('detections', [])
        
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
            print(f"Error parsing detection: {e}")
            continue

    # Check if we have valid detections after filtering
    if not valid_boxes:
        warnings.append("No persons or animals detected above confidence thresholds.")
        return IdentifyResponse(detections=[], warnings=warnings)

    # Count detection types for comprehensive reporting
    person_count = sum(1 for _, _, label, _ in valid_boxes if label == "person")
    animal_count = sum(1 for _, _, label, _ in valid_boxes if label == "animal")
    
    print(f"\n📊 MegaDetector Results: {len(valid_boxes)} total detections")
    print(f"   👤 Persons: {person_count} | 🦁 Animals: {animal_count}")
    
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
            print(f"⚠️  CRITICAL: Person detected [{i+1}] with {det_conf:.2%} confidence")
            continue
        

        # For animals ONLY - run BioCLIP species classification
        # NOTE: ALL ANIMALS ARE PROCESSED REGARDLESS OF POACHER PRESENCE
        if label == "animal":
            crop = crop_pil(pil_img, box)  # Crop from ORIGINAL color image, not processed
            
            try:
                result = classify_with_bioclip(
                    crop, 
                    allowed_species=allowed_species_list, 
                    confidence_threshold=species_confidence_threshold, 
                    topk=5
                )
                
                if result is not None:
                    predicted_species, species_conf, all_topk = result
                    
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
                    print(f"🦁 Animal detected [{i+1}]: {predicted_species} ({species_conf:.2%} confidence)")
                else:
                    # Below confidence threshold - still report as unclassified animal
                    det = Detection(
                        bbox=[x1, y1, x2, y2],
                        label="animal",
                        detector_confidence=det_conf,
                        species="UNKNOWN_ANIMAL",
                        species_confidence=0.0,
                        mask_png_base64=mask_to_base64_png(sam_masks_by_box[i]) if i in sam_masks_by_box else None
                    )
                    animal_detections.append(det)
                    print(f"🦁 Animal detected [{i+1}]: UNKNOWN_ANIMAL (low confidence)")
                    warnings.append(
                        f"Animal detected by MegaDetector but BioCLIP species confidence too low "
                        f"(< {species_confidence_threshold}). Classified as UNKNOWN_ANIMAL."
                    )
            
            except Exception as e:
                warnings.append(f"BioCLIP classification failed for animal detection {i}: {e}")
                print(f"❌ Animal detection [{i+1}] classification error: {e}")
                # Still add detection as unknown animal - NO ANIMAL MISSED!
                det = Detection(
                    bbox=[x1, y1, x2, y2],
                    label="animal",
                    detector_confidence=det_conf,
                    species="CLASSIFICATION_FAILED",
                    species_confidence=0.0,
                    mask_png_base64=mask_to_base64_png(sam_masks_by_box[i]) if i in sam_masks_by_box else None
                )
                animal_detections.append(det)

    # Combine detections in priority order: POACHERS FIRST, then animals
    detections_out = poacher_detections + animal_detections
    
    # Log final summary
    print(f"\n✅ Processing complete: {len(poacher_detections)} poacher(s), "
          f"{len(animal_detections)} animal(s)")
    
    return IdentifyResponse(detections=detections_out, warnings=warnings)


# -------------------------
# Server startup
# -------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting BioCLIP Poacher Detection Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)