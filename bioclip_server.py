# bioclip_server_v2.py
import io
import base64
import json
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
import open_clip

# Optional model imports (install if you use those features)
# pip install ultralytics segment-anything
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    # segment-anything imports
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
# Config / Candidate labels
# -------------------------
candidate_species = [
    # Mammals
    "Barking Deer", "Bengal Tiger", "Common Langur", "Four-Horned Antelope",
    "Indian Bison", "Indian Elephant", "Indian Giant Squirrel", "Indian Leopard",
    "Jungle Cat", "Sambar Deer", "Wild Boar", "Sloth Bear", "Indian Wolf",
    "Indian Fox", "Dhole", "Nilgai", "Spotted Deer", "Mouse Deer", "Blackbuck",
    "Rhesus Macaque", "Bonnet Macaque", "Lion-Tailed Macaque", "Chimpanzee",
    "Orangutan", "Gaur",
    # Birds
    "Hill Myna", "Hornbill", "Peacock", "Great Indian Bustard", "Forest Owl",
    "Kingfisher", "Parakeet", "Sandpiper", "Cattle Egret", "Indian Roller", "Shikra",
    # Reptiles
    "Saltwater Crocodile", "Mugger Crocodile", "Gharial", "King Cobra",
    "Indian Python", "Water Monitor", "Indian Cobra", "Russell's Viper",
    "Common Krait", "Saw-Scaled Viper", "Rat Snake", "Wolf Snake", "Barkudia Insularis",
    # Amphibians
    "Asian Common Toad", "Painted Globular Frog", "Indian Bullfrog",
    "Indian Skittering Frog", "Purple Frog",
    # Fish
    "Wallago Attu", "Labeo Rohita", "Catla Catla", "Cirrhinus mrigala",
    "Labeo calbasu", "Snakehead Fish", "Catfish", "Mahseer", "Freshwater Eel",
    # Other Notable Animals
    "Olive Ridley Sea Turtle", "Irrawaddy Dolphin"
]

# -------------------------
# Load BioCLIP model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# You must have open_clip & the model available; uses same call you had.
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
model = model.to(device)
model.eval()

text_tokens = tokenizer(candidate_species).to(device)

# -------------------------
# Optional: YOLO object detector
# -------------------------
DETECTOR = None
if YOLO_AVAILABLE:
    # choose a YOLOv8 model file or string. 'yolov8n.pt' is small and fast.
    # For best accuracy in wildlife + person detection use a larger model or custom weights.
    try:
        DETECTOR = YOLO("yolov8n.pt")  # fallback to model name (will download if needed)
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
    # You need to supply a SAM checkpoint or select a model from registry supported in your segment-anything installation.
    try:
        sam_checkpoint = "models/sam_vit_b_01ec64.pth"
        sam_model_type = "vit_b"
        SAM = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(device)
        SAM_MASK_GENERATOR = SamAutomaticMaskGenerator(SAM)
        print("SAM loaded.")
    except Exception as e:
        print("SAM not loaded. Provide the checkpoint and confirm segment-anything installation.", e)
else:
    print("segment-anything not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

# -------------------------
# Helpers
# -------------------------
def pil_from_bytes(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return img

def crop_pil(img: Image.Image, box: List[float]) -> Image.Image:
    # box: [x1, y1, x2, y2]
    x1, y1, x2, y2 = [int(round(x)) for x in box]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.width)
    y2 = min(y2, img.height)
    return img.crop((x1, y1, x2, y2))

def mask_to_base64_png(mask: np.ndarray) -> str:
    """
    mask: boolean or 0/1 2D numpy array. Returns base64-encoded PNG.
    """
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def classify_crop_bioclip(pil_crop: Image.Image, topk: int = 1):
    """
    Classify crop with BioCLIP. Returns list of (species, confidence) of length topk.
    """
    img_t = preprocess_val(pil_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img_t)
        text_features = model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)  # probabilities
    topv, topi = logits[0].topk(topk)
    results = []
    for score, idx in zip(topv.tolist(), topi.tolist()):
        results.append((candidate_species[idx], float(score)))
    return results

# -------------------------
# Response models
# -------------------------
class Detection(BaseModel):
    bbox: List[float]              # [x1,y1,x2,y2] in image coords
    label: str                     # "person" or "animal"
    detector_confidence: float
    species: Optional[str] = None  # filled for animals
    species_confidence: Optional[float] = None
    mask_png_base64: Optional[str] = None  # data URL
    extra: Optional[dict] = None

class IdentifyResponse(BaseModel):
    detections: List[Detection]
    warnings: Optional[List[str]] = None

# -------------------------
# Main endpoint
# -------------------------
@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    file: UploadFile = File(...),
    run_sam: bool = Form(False),           # whether to run SAM masks (can be slow)
    detector_threshold: float = Form(0.35),# min conf for detections
    topk_species: int = Form(1),           # top-k species to return per crop
):
    """
    Upload an image (multipart form-data, field name 'file').
    Optional form fields:
      - run_sam (bool): whether to run SAM mask generator for precise masks.
      - detector_threshold (float): minimum detector confidence to keep a detection (0-1)
      - topk_species (int): return top-k species per animal crop
    """
    img_bytes = await file.read()
    pil_img = pil_from_bytes(img_bytes)
    img_w, img_h = pil_img.size

    warnings = []
    detections_out: List[Detection] = []

    # If no detector available, fallback: run whole-image classification (single prediction)
    if DETECTOR is None:
        warnings.append("Object detector not available. Falling back to whole-image classification.")
        # classify whole image with BioCLIP
        species_preds = classify_crop_bioclip(pil_img, topk=topk_species)
        for species, species_conf in species_preds:
            det = Detection(
                bbox=[0.0, 0.0, float(img_w), float(img_h)],
                label="animal",
                detector_confidence=1.0,
                species=species,
                species_confidence=species_conf,
                mask_png_base64=None
            )
            detections_out.append(det)
        return IdentifyResponse(detections=detections_out, warnings=warnings)

    # Run detector
    # ultralytics YOLO returns a Results object. We pass numpy array or PIL.
    results = DETECTOR(pil_img)  # default size handling done by yolov8
    # results may be a list; take first
    results0 = results[0]

    # results0.boxes.xyxy, results0.boxes.conf, results0.boxes.cls
    boxes = []
    try:
        xyxy = results0.boxes.xyxy.cpu().numpy()  # shape (N,4)
        confs = results0.boxes.conf.cpu().numpy()
        cls_ids = results0.boxes.cls.cpu().numpy().astype(int)
    except Exception:
        # Fallback: sometimes API differs, try results0.boxes.data
        try:
            data = results0.boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,cls]
            xyxy = data[:, :4]
            confs = data[:, 4]
            cls_ids = data[:, 5].astype(int)
        except Exception as e:
            warnings.append(f"Unable to parse detector output: {e}")
            xyxy, confs, cls_ids = np.zeros((0,4)), np.array([]), np.array([])

    # ultralytics uses COCO class ids; id for person is 0 normally. We'll map COCO animal classes -> "animal"
    COCO_PERSON_CLASS_ID = 0  # typical for COCO
    # A simple set of COCO animal class ids (common ones). You can expand or use class names from model.names
    COCO_ANIMAL_CLASS_IDS = set([
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28  # cat/dog/horse/sheep/cow/elephant/bear/zebra/giraffe etc
    ])
    # If the model has .names use that to detect 'person' and animals by name matching
    model_names = getattr(DETECTOR, "model", None)
    # Better: use results0.names if provided
    names_map = getattr(results0, "names", None)

    for box, conf, cls_id in zip(xyxy, confs, cls_ids):
        if conf < detector_threshold:
            continue
        # Determine if label is person or animal
        label = "unknown"
        if names_map is not None:
            name = names_map.get(int(cls_id), str(cls_id)).lower()
            if "person" in name or "human" in name:
                label = "person"
            elif any(k in name for k in ["dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "deer", "tiger", "leopard", "monkey", "ape", "bird", "snake", "crocodile", "lizard", "fish", "turtle"]):
                label = "animal"
            else:
                label = name
        else:
            if int(cls_id) == COCO_PERSON_CLASS_ID:
                label = "person"
            elif int(cls_id) in COCO_ANIMAL_CLASS_IDS:
                label = "animal"
            else:
                label = f"class_{int(cls_id)}"

        boxes.append((box.tolist(), float(conf), label))

    # If no boxes found, optionally run whole-image classification to attempt detection
    if not boxes:
        warnings.append("No detections above threshold. Running whole-image classification fallback.")
        species_preds = classify_crop_bioclip(pil_img, topk=topk_species)
        for species, species_conf in species_preds:
            det = Detection(
                bbox=[0.0, 0.0, float(img_w), float(img_h)],
                label="animal",
                detector_confidence=1.0,
                species=species,
                species_confidence=species_conf,
                mask_png_base64=None
            )
            detections_out.append(det)
        return IdentifyResponse(detections=detections_out, warnings=warnings)

    # Optionally compute masks using SAM for each detected box
    sam_masks_by_box = {}
    if run_sam and SAM_MASK_GENERATOR is not None:
        # SAM expects an image as numpy array HWC, RGB
        arr = np.array(pil_img)
        all_masks = SAM_MASK_GENERATOR.generate(arr)
        # each mask has 'segmentation' (boolean mask) and 'bbox' etc
        # For simplicity, for each detector box we'll find the mask with largest IoU / overlap
        def iou_mask_box(mask, box):
            # compute IoU between boolean mask bbox and box
            ys, xs = np.where(mask)
            if len(xs) == 0:
                return 0.0
            mx1, my1 = xs.min(), ys.min()
            mx2, my2 = xs.max(), ys.max()
            ix1 = max(mx1, int(round(box[0])))
            iy1 = max(my1, int(round(box[1])))
            ix2 = min(mx2, int(round(box[2])))
            iy2 = min(my2, int(round(box[3])))
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            return 1.0  # coarse: since selecting by overlap we can be simple

        for i, (box, conf, label) in enumerate(boxes):
            # find mask with maximum overlap with box
            best_mask = None
            best_score = -1.0
            for m in all_masks:
                mask_bool = m["segmentation"]
                # crude overlap measure: bounding boxes overlap
                # we just select first mask whose bbox overlaps
                mbbox = m["bbox"]  # [x, y, w, h]
                mbx1, mby1, mbw, mbh = mbbox
                mbx2 = mbx1 + mbw
                mby2 = mby1 + mbh
                # overlap check
                if not (mbx2 < box[0] or mbx1 > box[2] or mby2 < box[1] or mby1 > box[3]):
                    best_mask = mask_bool
                    break
            if best_mask is not None:
                sam_masks_by_box[i] = best_mask

    # For each detected box, crop and classify if it's an animal
    for i, (box, det_conf, label) in enumerate(boxes):
        x1, y1, x2, y2 = box
        crop = crop_pil(pil_img, box)
        species = None
        species_conf = None

        if label == "person" or "person" in str(label).lower():
            # for humans we won't run BioCLIP (not in candidate list), but mark them
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

        # For animals (or unknown) run BioCLIP classification
        try:
            preds = classify_crop_bioclip(crop, topk=topk_species)
            # pick top1 for species fields, but include extra if topk>1
            if preds:
                species, species_conf = preds[0]
            extra = {"topk": [{"species": s, "confidence": c} for s, c in preds]} if len(preds) > 1 else None
        except Exception as e:
            warnings.append(f"BioCLIP classification failed for box {i}: {e}")
            species, species_conf, extra = None, None, None

        det = Detection(
            bbox=[x1, y1, x2, y2],
            label=label,
            detector_confidence=det_conf,
            species=species,
            species_confidence=species_conf,
            mask_png_base64=mask_to_base64_png(sam_masks_by_box[i]) if i in sam_masks_by_box else None,
            extra=extra
        )
        detections_out.append(det)

    return IdentifyResponse(detections=detections_out, warnings=warnings)
