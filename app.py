import numpy as np
from pathlib import Path
import uuid

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template, url_for

# -------------------------------------------------
# Project paths
# -------------------------------------------------
ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_resnet50_finetuned.pt"
UPLOAD_DIR = ROOT / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Flask app
# -------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(ROOT),
    static_folder=str(ROOT / "static")
)


# -------------------------------------------------
# Device
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Image preprocessing
# Must match validation/test preprocessing from training
# -------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# -------------------------------------------------
# Build same ResNet-50 architecture
# -------------------------------------------------
def build_model(num_classes: int):
    model = models.resnet50(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(in_features, num_classes)
    )

    return model


# -------------------------------------------------
# Load model once when app starts
# -------------------------------------------------
def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(DEVICE)
    model.eval()

    return model, classes, checkpoint


model, classes, checkpoint = load_model()


def date_like_precheck(image_path: Path):
    """
    Simple input-quality/date-likeness precheck.

    This is not deep learning.
    It is a lightweight heuristic based on the dataset style:

    - controlled white background
    - brown/orange date-like object
    - object occupies a reasonable portion of the image

    It helps reject obvious non-date images such as people, cars, code screenshots, etc.
    """

    image = Image.open(image_path).convert("RGB").resize((256, 256))
    arr = np.asarray(image).astype(np.float32) / 255.0

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)

    # Detect near-white background pixels.
    white_mask = (
        (r > 0.78) &
        (g > 0.78) &
        (b > 0.78) &
        ((max_channel - min_channel) < 0.18)
    )

    object_mask = ~white_mask

    white_background_ratio = white_mask.mean()
    object_ratio = object_mask.mean()

    object_pixels = object_mask.sum()

    if object_pixels == 0:
        date_color_ratio = 0.0
    else:
        # Detect rough brown/orange/dark-brown date-like colors.
        brown_mask = (
            (r > 0.12) &
            (r >= g * 0.85) &
            (g >= b * 0.70) &
            ((r - b) > 0.04)
        )

        date_color_ratio = (brown_mask & object_mask).sum() / object_pixels

    # These thresholds are intentionally simple and dataset-specific.
    passed = (
        white_background_ratio >= 0.35 and
        0.015 <= object_ratio <= 0.65 and
        date_color_ratio >= 0.18
    )

    if not passed:
        reason = "Image does not match the controlled date-image style."
    else:
        reason = "Image looks date-like under the current heuristic."

    return {
        "passed": bool(passed),
        "white_background_ratio": float(white_background_ratio),
        "object_ratio": float(object_ratio),
        "date_color_ratio": float(date_color_ratio),
        "reason": reason
    }


# -------------------------------------------------
# Prediction function
# -------------------------------------------------
@torch.no_grad()
def predict_image(image_path: Path, threshold: float = 0.70, top_k: int = 3):
    image = Image.open(image_path).convert("RGB")

    # First: simple date-like precheck
    precheck = date_like_precheck(image_path)

    image_tensor = inference_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)

    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

    top_probs = top_probs.squeeze(0).cpu().tolist()
    top_indices = top_indices.squeeze(0).cpu().tolist()

    best_index = top_indices[0]
    best_class = classes[best_index]
    best_confidence = top_probs[0]

    if not precheck["passed"]:
        decision_type = "not_date_like"
        decision_text = "REJECTED / NOT DATE-LIKE"
        accepted = False

    elif best_confidence < threshold:
        decision_type = "uncertain_date_class"
        decision_text = "DATE-LIKE BUT CLASS UNCERTAIN"
        accepted = False

    else:
        decision_type = "accepted"
        decision_text = "ACCEPTED"
        accepted = True

    top_predictions = []

    for index, prob in zip(top_indices, top_probs):
        top_predictions.append({
            "class_name": classes[index],
            "confidence": prob,
            "confidence_percent": prob * 100
        })

    return {
        "predicted_class": best_class,
        "confidence": best_confidence,
        "confidence_percent": best_confidence * 100,
        "accepted": accepted,
        "decision_type": decision_type,
        "decision_text": decision_text,
        "threshold": threshold,
        "threshold_percent": threshold * 100,
        "top_predictions": top_predictions,
        "precheck": precheck
    }


# -------------------------------------------------
# Home route
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None
    error = None

    default_threshold = 0.70
    default_top_k = 3

    if request.method == "POST":
        try:
            threshold = float(request.form.get("threshold", default_threshold))
            top_k = int(request.form.get("top_k", default_top_k))

            threshold = max(0.0, min(1.0, threshold))
            top_k = max(1, min(len(classes), top_k))

            uploaded_file = request.files.get("image")

            if uploaded_file is None or uploaded_file.filename == "":
                error = "Please upload an image."
            else:
                file_extension = Path(uploaded_file.filename).suffix.lower()

                if file_extension not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                    error = "Unsupported file type. Please upload JPG, PNG, WEBP, or BMP."
                else:
                    unique_name = f"{uuid.uuid4().hex}{file_extension}"
                    save_path = UPLOAD_DIR / unique_name

                    uploaded_file.save(save_path)

                    result = predict_image(
                        image_path=save_path,
                        threshold=threshold,
                        top_k=top_k
                    )

                    image_url = url_for("static", filename=f"uploads/{unique_name}")

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        error=error,
        classes=classes,
        device=str(DEVICE),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        checkpoint_stage=checkpoint.get("training_stage", "unknown"),
        val_accuracy=checkpoint.get("val_accuracy", "unknown"),
        default_threshold=default_threshold,
        default_top_k=default_top_k
    )


if __name__ == "__main__":
    print("Using device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Classes:", classes)

    app.run(debug=False)