import glob
import os

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageOps

CLASSES = ["lake", "ocean"]
EMOJI = {"lake": "🏞️", "ocean": "🌊"}
IMG_SIZE = (224, 224)
HERE = os.path.dirname(os.path.abspath(__file__))


def find_file(*candidates):
    """Return the first path that exists (works whether files are in subfolders or all at the top)."""
    for c in candidates:
        p = os.path.join(HERE, c)
        if os.path.exists(p):
            return p
    return None


st.set_page_config(page_title="Lake vs Ocean Classifier", page_icon="🌊", layout="centered")


@st.cache_resource
def load_model():
    path = find_file("model.onnx", "model/model.onnx")
    if path is None:
        st.error("Can't find `model.onnx` in the repository. Upload it next to `streamlit_app.py`.")
        st.stop()
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def predict(img: Image.Image) -> dict:
    """Same preprocessing as training: upright, RGB, resize to 224x224, pixels 0-255 (the model rescales itself)."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    img = img.convert("RGB").resize(IMG_SIZE[::-1], Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)[None]
    probs = load_model().run(None, {"image": x})[0][0]
    return {c: float(p) for c, p in zip(CLASSES, probs)}


def show_result(img: Image.Image, caption: str):
    probs = predict(img)
    best = max(probs, key=probs.get)
    conf = probs[best] * 100

    col_img, col_res = st.columns([1, 1], gap="large")
    with col_img:
        st.image(img, caption=caption, width="stretch")
    with col_res:
        st.subheader(f"{EMOJI[best]} {best.capitalize()}")
        if conf >= 80:
            st.write(f"The model is **{conf:.1f}%** sure this is {'a lake' if best == 'lake' else 'an ocean'}.")
        else:
            st.warning(f"Not very sure ({conf:.1f}%). This photo looks a bit like both.")
        for c in CLASSES:
            pct = probs[c] * 100
            label = "<0.01%" if 0 < pct < 0.01 else (f"{pct:.2f}%" if pct < 1 else f"{pct:.1f}%")
            st.write(f"**{c.capitalize()}**: {label}")
            st.progress(min(max(probs[c], 0.0), 1.0))


# ---------------- Page ----------------
st.title("🌊 Lake vs Ocean Classifier")
st.caption("Machine Learning Classification Project · KMITL · CNN (MobileNetV2 transfer learning)")
st.write(
    "Upload a photo and the model says whether it shows a **lake** or an **ocean**, "
    "with the percentage for each class."
)

tab_upload, tab_examples, tab_about = st.tabs(["📷 Upload a photo", "🖼️ Test images", "📊 About the model"])

with tab_upload:
    files = st.file_uploader(
        "Choose one or more photos", type=["jpg", "jpeg", "png", "webp", "bmp"], accept_multiple_files=True
    )
    if not files:
        st.info("Upload a photo to see the prediction.")
    for f in files or []:
        try:
            show_result(Image.open(f), f.name)
        except Exception:
            st.error(f"Couldn't read **{f.name}**. Try a JPG or PNG photo.")
        st.divider()

with tab_examples:
    examples = sorted(
        set(glob.glob(os.path.join(HERE, "*_test_*.jpg")) + glob.glob(os.path.join(HERE, "examples", "*.jpg")))
    )
    if not examples:
        st.info("No test images found in the repository.")
    else:
        names = [os.path.basename(p) for p in examples]
        choice = st.selectbox("Choose a test image", names)
        show_result(Image.open(examples[names.index(choice)]), choice)

        with st.expander("Classify all test images"):
            rows, correct = [], 0
            for p in examples:
                name = os.path.basename(p)
                probs = predict(Image.open(p))
                pred = max(probs, key=probs.get)
                true = next((c for c in CLASSES if name.lower().startswith(c)), "?")
                correct += pred == true
                rows.append({
                    "image": name, "true": true, "predicted": pred,
                    "lake %": round(probs["lake"] * 100, 2), "ocean %": round(probs["ocean"] * 100, 2),
                    "correct": "✅" if pred == true else "❌",
                })
            st.dataframe(rows, hide_index=True, width="stretch")
            st.metric("Test accuracy", f"{correct / len(rows):.0%}", f"{correct}/{len(rows)} correct", delta_color="off")

with tab_about:
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", "100%", "8 / 8", delta_color="off")
    c2.metric("Validation accuracy", "100%", "22 / 22", delta_color="off")
    c3.metric("Training images", "90", "+ 22 validation", delta_color="off")

    st.markdown(
        """
| Item | Setting |
|---|---|
| Input | 224 × 224 × 3 RGB |
| Data augmentation | Horizontal flip, rotation ±18°, zoom ±10%, shift ±5% (training only) |
| CNN base | MobileNetV2, 154 layers, ImageNet weights; top 30 layers fine-tuned |
| Classifier head | Global Average Pooling (1280) → Dropout 0.4 → Dense **128** ReLU → Dropout 0.4 → Dense **2** softmax |
| Parameters | 2,422,210 |
| Optimizer | Adam, lr 0.001 (stage 1), 0.00001 (stage 2) |
| Loss | Sparse categorical cross-entropy |
| Epochs / batch size | 30 + 15 (early stopping, patience 6) / 16 |
| Data split | 90 train · 22 validation · 8 test |
"""
    )
    for fname, cap in [
        ("training_curves.png", "Accuracy and loss for each epoch (dashed line = start of fine-tuning)"),
        ("confusion_validation.png", "Validation confusion matrix"),
        ("test_predictions.png", "Predictions on the 8 test images"),
    ]:
        p = find_file(fname, os.path.join("figures", fname))
        if p:
            st.image(p, caption=cap, width="stretch")
