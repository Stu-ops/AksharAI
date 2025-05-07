from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
import os
import io
import shutil
import tempfile
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import uvicorn

# Import py_text_scan (will be handled gracefully if missing)
try:
    import py_text_scan
    PY_TEXT_SCAN_AVAILABLE = True
except ImportError:
    PY_TEXT_SCAN_AVAILABLE = False
    print("Warning: py_text_scan module not available. OCR functionality will be limited.")

# Define paths to local resources
MODEL_PATH = "models/hindi_ocr_model.keras"
ENCODER_PATH = "models/label_encoder.pkl"
FONT_PATH = "Fonts/NotoSansDevanagari-Regular.ttf"

class OCRResponse(BaseModel):
    OCR_output: str
    word_count: int
    prediction_label: str


# --- FastAPI App Setup ---
app = FastAPI(
    title="Hindi OCR API",
    description="FASTAPI for Hindi Optical character recognition(OCR) using TensorFlow/Pytorch and OpenCV",
    version="1.0.0"
)

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found in local directory")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def load_label_encoder():
    if not os.path.exists(ENCODER_PATH):
        print(f"Error: Encoder file {ENCODER_PATH} not found in local directory")
        return None
    with open(ENCODER_PATH, 'rb') as f:
        return pickle.load(f)

model = None
label_encoder = None
session_files = {}

@app.on_event("startup")
async def startup_event():
    global model, label_encoder
    
    # Load model and encoder from local paths
    model = load_model()
    label_encoder = load_label_encoder()
    
    # Configure font if available
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.family'] = 'Noto Sans Devanagari'
    else:
        print(f"Warning: Font file {FONT_PATH} not found in local directory")
@app.get("/")
async def root():
    return HTMLResponse(content="""
    <html>
    <head>
        <title>Hindi OCR API</title>
    </head>
    <body>
        <h1>Hindi OCR API</h1>
        <p>Sample image with performed OCR.</p>
        <img src="/sample-image" alt="Sample Hindi OCR Text" width="800" height="400">
    </body>
    </html>
    """ ,status_code=200)
@app.get("/sample-image")
async def get_sample_image():
    return FileResponse("Sample_OCR_image.png", media_type="image/png")
def detect_words(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    word_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            cv2.rectangle(word_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            word_count += 1
    return word_img, word_count

def run_py_text_scan(image_path):
    if not PY_TEXT_SCAN_AVAILABLE:
        return "OCR text extraction unavailable - py_text_scan module not loaded"
    
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        py_text_scan.generate(image_path)
    except Exception as e:
        return f"Error running OCR: {str(e)}"
    finally:
        sys.stdout = old_stdout
    return buffer.getvalue()

def process_image(image_array):
    img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # --- Word Detection ---
    word_detected_img, word_count = detect_words(img)
    word_fd, word_path = tempfile.mkstemp(suffix=".png")
    os.close(word_fd)
    cv2.imwrite(word_path, word_detected_img)
    session_files['word_detection'] = word_path

    # --- Model Prediction ---
    pred_path = None
    try:
        img_resized = cv2.resize(img, (128, 32))
        img_norm = img_resized / 255.0
        img_input = img_norm[np.newaxis, ..., np.newaxis]

        if model is not None and label_encoder is not None:
            pred = model.predict(img_input)
            pred_label_idx = np.argmax(pred)
            pred_label = label_encoder.inverse_transform([pred_label_idx])[0]

            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            ax.set_title(f"पूर्वानुमान: {pred_label}")  # Hindi for "Prediction:" due to font support.
            ax.axis('off')

            pred_fd, pred_path = tempfile.mkstemp(suffix=".png")
            os.close(pred_fd)
            plt.savefig(pred_path)
            plt.close()
            session_files['prediction'] = pred_path
        else:
            pred_label = "Model or encoder not loaded"
    except Exception as e:
        pred_label = f"Error: {str(e)}"

    # --- Py Text Scan OCR ---
    temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
    os.close(temp_fd)
    cv2.imwrite(temp_path, img)
    OCR_output = run_py_text_scan(temp_path)
    os.unlink(temp_path)

    return {
        "OCR_output": OCR_output,
        "word_detection_path": word_path,
        "word_count": word_count,
        "prediction_path": pred_path,
        "prediction_label": pred_label
    }


@app.post("/process/", response_model=OCRResponse)
async def process(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    for key, filepath in session_files.items():
        if os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except:
                pass
    session_files.clear()

    temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        with Image.open(temp_path) as image:
            image_array = np.array(image)
        result = process_image(image_array)
        return OCRResponse(
            OCR_output=result["OCR_output"], 
            word_count=result["word_count"],
            prediction_label=result["prediction_label"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        try:
            os.unlink(temp_path)
        except Exception as e:
            print(f"Warning: Failed to delete temp file: {e}")
@app.get("/word-detection/")
async def get_word_detection():
    if 'word_detection' not in session_files or not os.path.exists(session_files['word_detection']):
        raise HTTPException(status_code=404, detail="Word detection image not found")
    return FileResponse(session_files['word_detection'])

@app.get("/prediction/")
async def get_prediction():
    if 'prediction' not in session_files or not os.path.exists(session_files['prediction']):
        raise HTTPException(status_code=404, detail="Prediction image not found")
    return FileResponse(session_files['prediction'])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)