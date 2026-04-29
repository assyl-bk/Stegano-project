from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from demo import (
    BINARY_MODEL_PATH,
    DEFAULT_RESULTS_DIR,
    MULTICLASS_MODEL_PATH,
    SUPPORTED_TECHNIQUES,
    decode_message_from_bit_iter,
    encode_dct,
    encode_fft,
    encode_lsb,
    encode_ssb4,
    encode_ssbn,
    extract_message,
    extract_message_auto,
    load_image,
    load_model,
    predict_image,
    preprocess_image,
)

ROOT_DIR = Path(__file__).resolve().parent
LOCAL_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SAMPLE_IMAGES = sorted(
    [path for path in ROOT_DIR.iterdir() if path.suffix.lower() in LOCAL_IMAGE_EXTS]
)

TECHNIQUE_LABELS = {
    "lsb": "LSB",
    "ssb4": "SSB-4",
    "ssbn": "SSB-N",
    "dct": "DCT",
    "fft": "FFT",
}

st.set_page_config(
    page_title="Stego Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --bg-main: #060b17;
            --bg-panel: #0f172a;
            --bg-panel-soft: #111c33;
            --text-main: #e5e7eb;
            --text-soft: #94a3b8;
            --border: rgba(148, 163, 184, 0.28);
            --accent-a: #38bdf8;
            --accent-b: #22c55e;
        }
        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(56, 189, 248, 0.12), transparent 28%),
                radial-gradient(circle at 85% 0%, rgba(34, 197, 94, 0.12), transparent 24%),
                linear-gradient(180deg, #040814 0%, #060b17 58%, #040814 100%);
            color: var(--text-main);
        }
        .hero {
            padding: 1.3rem 1.4rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0b1222 0%, #0f172a 55%, #101a33 100%);
            color: white;
            border: 1px solid rgba(56, 189, 248, 0.25);
            box-shadow: 0 16px 42px rgba(0, 0, 0, 0.34);
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-size: 2.2rem;
            margin: 0;
            line-height: 1.05;
        }
        .hero p {
            margin: 0.55rem 0 0 0;
            color: rgba(229, 231, 235, 0.88);
            font-size: 0.98rem;
        }
        div[data-baseweb="tab-panel"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.96) 0%, rgba(17, 28, 51, 0.96) 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            color: var(--text-main);
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 12px 34px rgba(0, 0, 0, 0.28);
            margin-bottom: 1rem;
        }
        div[data-baseweb="tab-list"] {
            gap: 0.35rem;
            margin-bottom: 0.35rem;
        }
        button[data-baseweb="tab"] {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.26);
            border-radius: 10px;
            color: var(--text-main);
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            border-color: rgba(56, 189, 248, 0.65);
            box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.45);
            background: rgba(15, 23, 42, 1);
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
        .stMarkdown p, .stMarkdown li, label, .stMetricLabel {
            color: var(--text-main) !important;
        }
        .stCaption, .stMarkdown small {
            color: var(--text-soft) !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1222 0%, #101a33 100%);
            border-right: 1px solid rgba(56, 189, 248, 0.22);
        }
        [data-testid="stMetric"] {
            background: rgba(8, 15, 30, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 12px;
            padding: 0.5rem 0.65rem;
        }
        .stDataFrame, .stTable {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 12px;
        }
        [data-testid="stFileUploader"] section,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div {
            background: rgba(8, 15, 30, 0.7) !important;
            color: var(--text-main) !important;
            border-color: rgba(148, 163, 184, 0.35) !important;
        }
        [data-testid="stButton"] button,
        [data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, var(--accent-a) 0%, #0ea5e9 45%, var(--accent-b) 100%);
            color: #04101f;
            border: none;
            font-weight: 700;
            box-shadow: 0 8px 22px rgba(34, 197, 94, 0.24);
        }
        [data-testid="stButton"] button:hover,
        [data-testid="baseButton-primary"]:hover {
            filter: brightness(1.05);
        }
        .small-note {
            color: var(--text-soft);
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Stego Lab</h1>
        <p>Attack, detect, and recover messages from local steganography models in one interface.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource

def get_model_bundles():
    binary_bundle = load_model(BINARY_MODEL_PATH)
    multi_bundle = load_model(MULTICLASS_MODEL_PATH)
    return binary_bundle, multi_bundle


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def get_image_from_upload(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def get_sample_choices() -> list[str]:
    return [path.name for path in SAMPLE_IMAGES]


def resolve_sample(name: str) -> Path:
    for path in SAMPLE_IMAGES:
        if path.name == name:
            return path
    raise FileNotFoundError(name)


def attack_image(image: Image.Image, technique: str, message: str, resize_to: int, ssbn_n: int):
    cover = np.array(image.resize((resize_to, resize_to)), dtype=np.uint8)
    encoder_map = {
        "lsb": encode_lsb,
        "ssb4": encode_ssb4,
        "ssbn": lambda arr, msg: encode_ssbn(arr, msg, n=ssbn_n),
        "dct": encode_dct,
        "fft": encode_fft,
    }
    stego = encoder_map[technique](cover, message)
    return Image.fromarray(stego)


def detect_image(image: Image.Image, technique_hint: str | None):
    binary_bundle, multi_bundle = get_model_bundles()
    binary_model, binary_classes, binary_size = binary_bundle
    multi_model, multi_classes, multi_size = multi_bundle
    img_size = binary_size if binary_size == multi_size else min(binary_size, multi_size)
    binary_result = predict_image(binary_model, binary_classes, image, img_size)
    multi_result = predict_image(multi_model, multi_classes, image, img_size)
    preferred = technique_hint or multi_result["label"]
    message, decoded_from, tried = extract_message_auto(image, preferred, n=1)
    is_stego = binary_result["label"] != "clean"
    return {
        "is_stego": is_stego,
        "binary": binary_result,
        "multi": multi_result,
        "message": message,
        "decoded_from": decoded_from,
        "tried": tried,
    }


def technique_hint_option(label: str) -> str | None:
    return None if label == "Auto" else label


st.sidebar.markdown("## Controls")
st.sidebar.caption("Models are loaded from the local `.pth` files in this folder.")
st.sidebar.write(f"Binary model: `{BINARY_MODEL_PATH.name}`")
st.sidebar.write(f"Multi-class model: `{MULTICLASS_MODEL_PATH.name}`")
st.sidebar.write(f"Saved outputs: `{DEFAULT_RESULTS_DIR.name}/`")

if SAMPLE_IMAGES:
    st.sidebar.markdown("### Local samples")
    st.sidebar.write(
        ", ".join(path.name for path in SAMPLE_IMAGES if path.suffix.lower() in LOCAL_IMAGE_EXTS)
    )

attack_tab, detect_tab, eval_tab = st.tabs(["Attack", "Detect", "Evaluate"])

with attack_tab:
    st.subheader("Create a stego image")
    attack_source = st.radio("Image source", ["Upload image", "Local sample"], horizontal=True)
    attack_upload = None
    attack_sample = None
    if attack_source == "Upload image":
        attack_upload = st.file_uploader("Choose a cover image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"], key="attack_upload")
    else:
        options = get_sample_choices() or ["cat.jpg"]
        attack_sample = st.selectbox("Choose a local sample", options, key="attack_sample")

    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        attack_message = st.text_area("Hidden message", value="this is our cyber project", height=120)
    with col_b:
        attack_technique = st.selectbox("Technique", ["lsb", "ssb4", "ssbn", "dct", "fft"], format_func=lambda x: TECHNIQUE_LABELS[x])
        attack_size = st.slider("Resize before embedding", 64, 512, 224, step=8)
    with col_c:
        attack_n = st.number_input("SSB-N bit depth", min_value=1, max_value=4, value=1, step=1)
        attack_name = st.text_input("Output name", value="stego_output.png")

    attack_run = st.button("Generate stego image", type="primary")
    if attack_run:
        if attack_source == "Upload image":
            if attack_upload is None:
                st.warning("Upload a cover image first.")
            else:
                source_image = get_image_from_upload(attack_upload)
                stego_image = attack_image(source_image, attack_technique, attack_message, attack_size, int(attack_n))
                out_path = DEFAULT_RESULTS_DIR / attack_name
                stego_image.save(out_path)
                left, right = st.columns(2)
                with left:
                    st.image(source_image, caption="Cover image", use_container_width=True)
                with right:
                    st.image(stego_image, caption=f"Stego image ({TECHNIQUE_LABELS[attack_technique]})", use_container_width=True)
                st.success(f"Saved to {out_path}")
                st.download_button(
                    "Download stego PNG",
                    data=image_to_png_bytes(stego_image),
                    file_name=attack_name if attack_name.lower().endswith(".png") else f"{attack_name}.png",
                    mime="image/png",
                )
        else:
            sample_path = resolve_sample(attack_sample or "cat.jpg")
            source_image = Image.open(sample_path).convert("RGB")
            stego_image = attack_image(source_image, attack_technique, attack_message, attack_size, int(attack_n))
            out_path = DEFAULT_RESULTS_DIR / attack_name
            stego_image.save(out_path)
            left, right = st.columns(2)
            with left:
                st.image(source_image, caption=f"Cover sample: {sample_path.name}", use_container_width=True)
            with right:
                st.image(stego_image, caption=f"Stego image ({TECHNIQUE_LABELS[attack_technique]})", use_container_width=True)
            st.success(f"Saved to {out_path}")
            st.download_button(
                "Download stego PNG",
                data=image_to_png_bytes(stego_image),
                file_name=attack_name if attack_name.lower().endswith(".png") else f"{attack_name}.png",
                mime="image/png",
            )

with detect_tab:
    st.subheader("Detect and extract")
    detect_source = st.radio("Image source", ["Upload image", "Local sample"], horizontal=True, key="detect_source")
    detect_upload = None
    detect_sample = None
    if detect_source == "Upload image":
        detect_upload = st.file_uploader("Choose an image to analyze", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"], key="detect_upload")
    else:
        options = get_sample_choices() or ["cat_stego.png"]
        detect_sample = st.selectbox("Choose a local sample", options, key="detect_sample")

    hint_label = st.selectbox("Technique hint", ["Auto"] + SUPPORTED_TECHNIQUES, format_func=lambda x: "Auto" if x == "Auto" else TECHNIQUE_LABELS[x])
    detect_run = st.button("Run detection", type="primary", key="detect_run")

    if detect_run:
        if detect_source == "Upload image":
            if detect_upload is None:
                st.warning("Upload an image first.")
            else:
                image = get_image_from_upload(detect_upload)
                result = detect_image(image, technique_hint_option(hint_label))
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Binary", f"{result['binary']['label'].upper()}", f"{result['binary']['confidence']:.1%}")
                with c2:
                    st.metric("Multiclass", f"{result['multi']['label'].upper()}", f"{result['multi']['confidence']:.1%}")
                with c3:
                    st.metric("Decoded from", result["decoded_from"].upper() if result["decoded_from"] else "none")
                st.image(image, caption="Input image", use_container_width=True)
                st.write(f"**Extracted message:** {result['message'] or '(none)'}")
                st.write(f"**Fallback attempts:** {', '.join(result['tried'])}")
                st.json({"binary_probs": result["binary"]["all_probs"], "multi_probs": result["multi"]["all_probs"]})
        else:
            sample_path = resolve_sample(detect_sample or "cat_stego.png")
            image = Image.open(sample_path).convert("RGB")
            result = detect_image(image, technique_hint_option(hint_label))
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Binary", f"{result['binary']['label'].upper()}", f"{result['binary']['confidence']:.1%}")
            with c2:
                st.metric("Multiclass", f"{result['multi']['label'].upper()}", f"{result['multi']['confidence']:.1%}")
            with c3:
                st.metric("Decoded from", result["decoded_from"].upper() if result["decoded_from"] else "none")
            st.image(image, caption=f"Input sample: {sample_path.name}", use_container_width=True)
            st.write(f"**Extracted message:** {result['message'] or '(none)'}")
            st.write(f"**Fallback attempts:** {', '.join(result['tried'])}")
            st.json({"binary_probs": result["binary"]["all_probs"], "multi_probs": result["multi"]["all_probs"]})

with eval_tab:
    st.subheader("Batch evaluation")
    eval_uploads = st.file_uploader(
        "Upload one or more images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
        key="eval_uploads",
    )
    eval_hint_label = st.selectbox("Technique hint for batch decode", ["Auto"] + SUPPORTED_TECHNIQUES, format_func=lambda x: "Auto" if x == "Auto" else TECHNIQUE_LABELS[x], key="eval_hint")
    eval_run = st.button("Evaluate batch", type="primary", key="eval_run")

    if eval_run:
        if not eval_uploads:
            st.warning("Upload at least one image.")
        else:
            binary_counts = Counter()
            multi_counts = Counter()
            rows = []
            for uploaded_file in eval_uploads:
                image = get_image_from_upload(uploaded_file)
                result = detect_image(image, technique_hint_option(eval_hint_label))
                binary_counts[result["binary"]["label"]] += 1
                multi_counts[result["multi"]["label"]] += 1
                rows.append(
                    {
                        "file": uploaded_file.name,
                        "binary": result["binary"]["label"],
                        "binary_conf": round(result["binary"]["confidence"], 4),
                        "multi": result["multi"]["label"],
                        "multi_conf": round(result["multi"]["confidence"], 4),
                        "decoded_from": result["decoded_from"] or "none",
                        "message": result["message"] or "(none)",
                    }
                )
            st.write("**Summary counts**")
            st.json({"binary": dict(binary_counts), "multi": dict(multi_counts)})
            st.write("**Per-file results**")
            st.table(rows)

st.caption(
    "Detection uses the loaded binary and multiclass checkpoints from this folder. "
    "Extraction tries the hinted technique first, then falls back across all supported techniques."
)
