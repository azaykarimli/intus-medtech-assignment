from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance, UnidentifiedImageError
import numpy as np
import cv2
import io, base64

# --- Config ---
MAX_UPLOAD_MB = 10
ALLOWED_ORIGINS = [
    "https://azaykarimli.github.io",
    "https://azaykarimli.github.io/intus-medtech-assignment/",
]
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}

app = FastAPI(title="MedTech Image Phase Simulation")

# CORS: restrict to your GitHub Pages origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
    multipart/form-data:
      - file: PNG/JPG
      - phase: 'arterial' | 'venous'
    returns: { processed_image: <data-url> }
    """
    # Basic content-type check (helps catch PDFs etc.)
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported media type. Use JPG or PNG.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_MB} MB).")

    # Validate image with Pillow
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()  # validates integrity
        # re-open because verify() leaves the file in an unusable state
        img = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file. Use JPG or PNG.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Normalize mode
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    phase = (phase or "").lower().strip()
    if phase not in ("arterial", "venous"):
        raise HTTPException(status_code=400, detail="phase must be 'arterial' or 'venous'")

    # --- Processing ---
    if phase == "arterial":
        # Increased contrast
        img_proc = ImageEnhance.Contrast(img).enhance(1.6)
        if img_proc.mode == "RGB":
            img_proc = ImageEnhance.Color(img_proc).enhance(1.12)
    else:
        # Venous: gaussian smoothing
        arr = np.array(img)
        arr_blur = cv2.GaussianBlur(arr, (7, 7), 1.5)
        img_proc = Image.fromarray(arr_blur)

    return JSONResponse({"processed_image": pil_to_data_url(img_proc, "PNG")})
