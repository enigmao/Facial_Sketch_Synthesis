
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Load your Pix2Pix model here (expects a TorchScript file named 'pix2pix_model.pt' in project root)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = torch.jit.load("pix2pix_model.pt", map_location=device)
    model.eval()
    print("✅ Loaded pix2pix_model.pt")
except Exception as e:
    model = None
    print("⚠️ Warning: Pix2Pix model not found or failed to load:", e)
    print("    Place 'pix2pix_model.pt' in the project root to enable generation.")

# Image transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(input_path)

        # Run Pix2Pix model (or fallback)
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
        if model:
            img = Image.open(input_path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(img_t)
                # Some TorchScript pix2pix export returns a tuple; handle both tensor or (tensor, ...)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                output = pred[0].detach().cpu()
            out_img = to_pil(output.clamp(0, 1))
            out_img.save(output_path)
        else:
            # Fallback: just copy input to output
            Image.open(input_path).save(output_path)

        return redirect(url_for("result", filename=filename))
    return render_template("index.html")

@app.route("/result/<filename>")
def result(filename):
    # Simple page to show original + generated side-by-side
    return f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Result - Facial Sketch Synthesis</title>
        <style>
            body {{ font-family: Arial, sans-serif; background:#1e1e1e; color:#fff; margin:0; padding:40px; }}
            h1 {{ color:#f39c12; }}
            .grid {{ display:flex; gap:24px; align-items:flex-start; }}
            .card {{ background:#2c2c2c; border-radius:12px; padding:16px; box-shadow:0 4px 10px rgba(0,0,0,0.5); }}
            img {{ max-width:400px; height:auto; border-radius:8px; display:block; }}
            a.button {{ display:inline-block; margin-top:16px; padding:10px 16px; background:#f39c12; color:#1e1e1e; text-decoration:none; border-radius:6px; font-weight:bold; }}
            a.button:hover {{ background:#d35400; }}
        </style>
    </head>
    <body>
        <h1>Facial Sketch Synthesis - Result</h1>
        <div class="grid">
            <div class="card">
                <h3>Input Sketch</h3>
                <img src="/uploads/{filename}" alt="input">
            </div>
            <div class="card">
                <h3>Generated Face</h3>
                <img src="/outputs/{filename}" alt="output">
                <a class="button" href="/outputs/{filename}" download>Download Generated Image</a>
            </div>
        </div>
        <p><a class="button" href="/">← Try another image</a></p>
    </body>
    </html>
    """

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == "__main__":
    # On platforms like Render/Heroku, PORT env is provided
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
