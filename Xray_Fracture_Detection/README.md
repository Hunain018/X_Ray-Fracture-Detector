# 🛡️ Weapon Detection in X-Ray Images — YOLOv11

Fine-tuned YOLOv11 model for detecting weapons in X-ray baggage/security scan
images, served through a FastAPI backend with a Streamlit frontend.

- **Model:** YOLOv11 (Ultralytics), fine-tuned on the Weapon Detection (X-Ray
  Images) dataset (see training notebook `Untitled2.ipynb`).
- **Backend:** FastAPI — `/predict`, `/predict/image`, `/predict/base64` endpoints.
- **Frontend:** Streamlit with custom CSS — upload an image, see bounding
  boxes + confidence scores.

```
weapon-xray-detection/
├── backend/
│   ├── main.py            # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── app.py              # Streamlit app (custom CSS)
│   └── requirements.txt
├── model/
│   └── best.pt              # <-- put your trained weights here
├── Dockerfile               # runs backend + frontend in one container
├── start.sh
└── README.md
```

## 1. Add your trained model weights

Your training notebook saves the best checkpoint at:
```
Xray_runs/exp1/weights/best.pt
```
Download that file from Colab/wherever you trained and place it at:
```
model/best.pt
```

> **Note on GitHub file size:** `best.pt` for YOLOv11n/s is usually 5–50MB,
> which GitHub allows, but if you used a larger variant (m/l/x) and it's
> over 100MB, use [Git LFS](https://git-lfs.com/):
> ```bash
> git lfs install
> git lfs track "*.pt"
> git add .gitattributes model/best.pt
> ```
> Alternatively, upload the weights to the
> [Hugging Face Hub](https://huggingface.co/) and download them at container
> startup by setting `MODEL_PATH` to the downloaded path in `start.sh`.

## 2. Run locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend** (in a second terminal):
```bash
cd frontend
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8000   # Windows: set BACKEND_URL=http://localhost:8000
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## 3. Run with Docker (backend + frontend in one container)

```bash
docker build -t weapon-xray-app .
docker run -p 8501:8501 -e PORT=8501 weapon-xray-app
```

Open `http://localhost:8501`.

## 4. Deploy — recommended options

### Option A: Single link via Hugging Face Spaces (Docker) — easiest
1. Create a new Space → SDK: **Docker**.
2. Push this repo (including `model/best.pt`) to the Space's git remote.
3. The Space builds the `Dockerfile` and exposes the Streamlit UI on port 7860
   automatically — just set `ENV PORT=7860` in the Dockerfile (or leave the
   Space's default `PORT` env, HF Spaces sets it for you).
4. Your live link: `https://huggingface.co/spaces/<your-username>/<space-name>`

### Option B: Render.com (Docker Web Service)
1. New → **Web Service** → connect your GitHub repo.
2. Environment: **Docker** (it will detect the `Dockerfile`).
3. Render sets `$PORT` automatically — `start.sh` already respects it.
4. Deploy → you get a live `https://your-app.onrender.com` link.

### Option C: Separate services (backend on Render, frontend on Streamlit Cloud)
1. Deploy `backend/` as its own Render Web Service (Python, start command:
   `uvicorn main:app --host 0.0.0.0 --port $PORT`).
2. Deploy `frontend/` on [Streamlit Community Cloud](https://streamlit.io/cloud),
   setting the `BACKEND_URL` secret/env var to your Render backend URL.

## 5. API reference (backend)

| Method | Endpoint          | Description                                   |
|--------|-------------------|------------------------------------------------|
| GET    | `/`               | Health check + whether model is loaded         |
| GET    | `/classes`        | List of class names the model detects          |
| POST   | `/predict`        | Upload image → JSON detections                 |
| POST   | `/predict/image`  | Upload image → annotated PNG image             |
| POST   | `/predict/base64` | Upload image → JSON detections + base64 image  |

Example:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg" \
  -F "conf=0.5"
```

## 6. Submission checklist

- [ ] `model/best.pt` added (via Git LFS if >100MB)
- [ ] Pushed to a public GitHub repo
- [ ] Deployed (Hugging Face Spaces / Render / Streamlit Cloud)
- [ ] Live link + GitHub repo link submitted on the portal
