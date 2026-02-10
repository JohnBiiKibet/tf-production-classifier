"""
FastAPI application for the vision system with TFLite inference.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(title="Plant Vision System API", version="1.0.0")

# Model configuration
IMG_SIZE = 224
MODEL_PATH = '../models/plant_model_v1.tflite'

# Class labels mapping (update based on your dataset classes)
CLASS_LABELS = {
    0: "Healthy",
    1: "Diseased",
    2: "Damaged",
    3: "Unknown"
    # Add more classes as needed to match your NUM_CLASSES in train.py
}

# Global interpreter (loaded once)
interpreter = None
input_details = None
output_details = None

def load_tflite_model():
    """Load TFLite model on startup."""
    global interpreter, input_details, output_details
    
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: TFLite model not found at {MODEL_PATH}")
        print("Using Keras model instead. Run scripts/optimize.py to create TFLite model.")
        return False
    
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"✓ TFLite model loaded from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return False

def preprocess_image(image_bytes):
    """Convert image bytes to preprocessed tensor."""
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize to model input size
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize (for float models)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def run_inference(image_array):
    """Run TFLite inference."""
    try:
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], image_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        return output_data
    except Exception as e:
        raise RuntimeError(f"Inference error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_tflite_model()

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "message": "Plant Vision System API",
        "version": "1.0.0",
        "status": "Running",
        "model": "Plant Classification Model"
    }

@app.get("/health")
def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": interpreter is not None,
        "model_path": MODEL_PATH,
        "input_size": IMG_SIZE
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant class from uploaded image.
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        Predictions with class and confidence scores
    """
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read uploaded file
        image_bytes = await file.read()
        
        # Preprocess image
        img_array = preprocess_image(image_bytes)
        
        # Run inference
        predictions = run_inference(img_array)
        
        # Extract results
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        class_name = CLASS_LABELS.get(class_idx, f"Class_{class_idx}")
        all_scores = predictions[0].tolist()
        
        return JSONResponse({
            "class": class_name,
            "class_id": class_idx,
            "confidence": round(confidence, 4),
            "all_scores": [round(score, 4) for score in all_scores],
            "filename": file.filename,
            "status": "success"
        })
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """
    Batch predict multiple images.
    
    Args:
        files: List of image files
    
    Returns:
        List of predictions for each image
    """
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []

    for file in files:
        try:
            image_bytes = await file.read()
            img_array = preprocess_image(image_bytes)
            predictions = run_inference(img_array)

            class_idx = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            class_name = CLASS_LABELS.get(class_idx, f"Class_{class_idx}")

            results.append({
                "filename": file.filename,
                "class": class_name,
                "class_id": class_idx,
                "confidence": round(confidence, 4),
                "status": "success"
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
    
    return JSONResponse({
        "results": results,
        "total": len(files),
        "successful": sum(1 for r in results if r["status"] == "success")
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
