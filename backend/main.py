from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import io, base64

app = FastAPI(title="MedTech Image Phase Simulation")

# For demo: allow all. Later, restrict to your GitHub Pages origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

def pil_to_data_url(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

@app.post("/process")
async def process_image(file: UploadFile = File(...), phase: str = Form(...)):
    """
    multipart/form-data with:
      - file: PNG/JPG
      - phase: 'arterial' | 'venous'
    returns: { processed_image: <data-url> }
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    phase = (phase or "").lower().strip()
    if phase not in ("arterial", "venous"):
        raise HTTPException(status_code=400, detail="phase must be 'arterial' or 'venous'")

    if phase == "arterial":
        # increase contrast visibly
        img_proc = ImageEnhance.Contrast(img).enhance(1.6)
        if img_proc.mode == "RGB":
            img_proc = ImageEnhance.Color(img_proc).enhance(1.12)
    else:
        # venous: gaussian smoothing
        arr = np.array(img)
        arr_blur = cv2.GaussianBlur(arr, (7, 7), 1.5)
        img_proc = Image.fromarray(arr_blur)

    return JSONResponse({"processed_image": pil_to_data_url(img_proc, "PNG")})
