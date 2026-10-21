"""
FastAPI backend for the Fracture Detection (X-Ray) YOLOv11 project.

Endpoints:
    GET  /                -> health check
    GET  /classes          -> list of class names the model can detect
    POST /predict           -> upload an image, get back detections (JSON)
    POST /predict/image     -> upload an image, get back the annotated image (PNG)

Run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import os
import base64
import logging

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fracture-detector")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", r"D:\Ai and datascience\Xray_fracture_detection\backend\weights\best.onnx")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.5"))

app = FastAPI(
    title="X-Ray Fracture Detection API",
    description="YOLOv11 fine-tuned model served over a REST API for detecting weapons in X-ray images.",
    version="1.0.0",
)

# Allow the Streamlit frontend (or any client) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
model = None


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.warning(
            "Model weights not found at %s. "
            "Place your trained best.pt in the /model folder or set MODEL_PATH.",
            MODEL_PATH,
        )
        return
    logger.info("Loading YOLO model from %s ...", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    logger.info("Model loaded. Classes: %s", model.names)


def _ensure_model_loaded():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Expected weights at '{MODEL_PATH}'.",
        )


def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


@app.get("/classes")
def get_classes():
    _ensure_model_loaded()
    return {"classes": model.names}


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = CONF_THRESHOLD):
    """Run inference and return detections as JSON (boxes, classes, confidences)."""
    _ensure_model_loaded()

    contents = await file.read()
    image = _read_image(contents)

    results = model.predict(source=np.array(image), conf=conf, verbose=False)
    r = results[0]

    detections = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        detections.append(
            {
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": round(float(box.conf[0]), 4),
                "bbox_xyxy": [round(v, 2) for v in box.xyxy[0].tolist()],
            }
        )

    return JSONResponse(
        {
            "filename": file.filename,
            "num_detections": len(detections),
            "detections": detections,
            "fracture_detected": len(detections) > 0,
        }
    )


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), conf: float = CONF_THRESHOLD):
    """Run inference and return the annotated image (with bounding boxes drawn) as PNG."""
    _ensure_model_loaded()

    contents = await file.read()
    image = _read_image(contents)

    results = model.predict(source=np.array(image), conf=conf, verbose=False)
    r = results[0]

    annotated_bgr = r.plot()  # numpy array, BGR
    annotated_rgb = annotated_bgr[:, :, ::-1]
    annotated_img = Image.fromarray(annotated_rgb)

    buf = io.BytesIO()
    annotated_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.post("/predict/base64")
async def predict_base64(file: UploadFile = File(...), conf: float = CONF_THRESHOLD):
    """Convenience endpoint: returns both JSON detections AND the annotated image
    (base64-encoded) in a single response — handy for the Streamlit frontend."""
    _ensure_model_loaded()

    contents = await file.read()
    image = _read_image(contents)

    results = model.predict(source=np.array(image), conf=conf, verbose=False)
    r = results[0]

    detections = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        detections.append(
            {
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": round(float(box.conf[0]), 4),
                "bbox_xyxy": [round(v, 2) for v in box.xyxy[0].tolist()],
            }
        )

    annotated_bgr = r.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    annotated_img = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    annotated_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse(
        {
            "filename": file.filename,
            "num_detections": len(detections),
            "detections": detections,
            "fracture_detected": len(detections) > 0,
            "annotated_image_base64": encoded,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
