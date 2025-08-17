
# Facial Sketch Synthesis Web App (Pix2Pix)

Minimal Flask app to upload a sketch and generate a realistic face using a **Pix2Pix Conditional GAN**.

## How to run locally
```bash
pip install -r requirements.txt
python application.py
# open http://localhost:5000
```

## Deploy (Heroku/Render)
- Ensure `Procfile` and `requirements.txt` are committed.
- Add your `pix2pix_model.pt` TorchScript model file in the repo root.
- Deploy. The app will bind to `PORT` env automatically.

## Model file
Put `pix2pix_model.pt` at the repository root. The app will load it on startup.
