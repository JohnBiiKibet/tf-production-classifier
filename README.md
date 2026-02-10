# Edge-Optimized Production Vision System

Small, edge-friendly FastAPI service for plant image classification using MobileNetV2.

## Project Layout

- `app/` - FastAPI application (`app/main.py`)
- `data/` - Place image folders here (one folder per class)
- `models/` - Trained models and converted TFLite files
- `scripts/` - Training and optimization scripts
  - `scripts/train.py` - Train MobileNetV2 transfer model
  - `scripts/optimize.py` - Convert `.keras` -> `.tflite`
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration

## Quickstart (Windows PowerShell)

1. Create the virtual environment (already created in this repo as `venv`):

```powershell
python -m venv venv
```

2. Install dependencies into the `venv`:

```powershell
# Use the venv python to avoid execution policy issues
.\venv\Scripts\python -m pip install -r requirements.txt
```

3. Prepare your dataset under `data/` with this structure:

```
data/
├─ class_0/
│  ├─ img001.jpg
│  └─ img002.jpg
└─ class_1/
   ├─ img001.jpg
   └─ img002.jpg
```

4. Train the model:

```powershell
.\venv\Scripts\python scripts\train.py
```

This will save the Keras model to `models/plant_model_v1.keras`.

5. Convert to TFLite for edge deployment:

```powershell
.\venv\Scripts\python scripts\optimize.py
```

6. Run the FastAPI server (development):

```powershell
.\venv\Scripts\python app\main.py
```

Or with Uvicorn (recommended):

```powershell
.\venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /` - Health/info
- `GET /health` - Model status
- `POST /predict` - Single-file image prediction (multipart/form-data `file`)
- `POST /batch-predict` - Multiple images

Example curl:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@plant.jpg"
```

## Docker

Build and run the container:

```bash
docker build -t plant-vision-system .
docker run -p 80:80 plant-vision-system
```

## Notes

- Update `app/main.py` → `CLASS_LABELS` to match the classes in your dataset.
- The project includes data augmentation and MobileNetV2 transfer learning suitable for edge deployment.
- Large files such as `data/` and `models/` are in `.gitignore` by default; store large artifacts externally.

If you want, I can now:
- Verify the `venv` installs and re-run the training command,
- Start a smoke-test of the FastAPI server,
- Or push this repo to a remote (I will need the remote URL).
