from flask import Flask, render_template, request, send_file, redirect, url_for, session, jsonify
import os, csv, datetime, json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import cv2

# ------------------ Firebase Setup ------------------
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ------------------ Flask App Setup ------------------
from routes.register import register_bp
from routes.login import login_bp  # login blueprint

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'fallback_dev_key')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB limit

# Register Blueprints
app.register_blueprint(register_bp)
app.register_blueprint(login_bp)

# ------------------ Default Routes ------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/index")
def index():
    if session.get("user_email"):
        return render_template("index.html")
    return redirect(url_for("login.login"))

# ------------------ ML Setup ------------------
UPLOAD_FOLDER = "uploads"
LOG_FOLDER = "logs"
MODEL_PATH = "models/child_labour_model.pth"
LABELS_PATH = "models/labels.json"
LOG_FILE = os.path.join(LOG_FOLDER, "detection_logs.csv")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

with open(LABELS_PATH, "r") as f:
    idx_to_class = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Updated to avoid pretrained warning
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------ Detection Function ------------------
def predict_video(video_path, sample_frames=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(frame_count // sample_frames, 1)
    child_detected = False

    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)
            label = idx_to_class[str(pred.item())]

            if label.lower() == "child" and confidence.item() > 0.7:
                child_detected = True
                break  # no need to check further frames

    cap.release()

    if child_detected:
        return {"status": "alert", "message": "⚠️ ALERT: Child detected in workplace"}
    else:
        return {"status": "safe", "message": "✅ Only Adults detected"}

# ------------------ Video Upload Route ------------------
@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video. Unsupported format or corrupted file."}), 400
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 0:
            return jsonify({"error": "Video has zero frames."}), 400
        cap.release()

        # Run ML detection
        result = predict_video(filepath)

        # Log CSV
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([file.filename, result["message"],
                             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        # Log Firestore
        db.collection("detection_logs").add({
            "filename": file.filename,
            "result": result["message"],
            "timestamp": datetime.datetime.now()
        })

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] Failed to process video: {e}")
        return jsonify({"error": f"Error occurred while processing the video: {str(e)}"}), 500

# ------------------ Logs Download Route ------------------
@app.route("/logs", methods=["GET"])
def get_logs():
    return send_file(LOG_FILE, as_attachment=True)

# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)
