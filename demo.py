import argparse
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

CLASS_NAMES = ["clean", "lsb", "ssb4", "ssbn", "dct", "fft"]
SUPPORTED_TECHNIQUES = ["lsb", "ssb4", "ssbn", "dct", "fft"]
DELIMITER = "<<END>>"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parent
BINARY_MODEL_PATH = ROOT_DIR / "binary_model.pth"
MULTICLASS_MODEL_PATH = ROOT_DIR / "multiclass_model.pth"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_RESULTS_DIR.mkdir(exist_ok=True)


def build_model(num_classes):
    model = models.efficientnet_b2(weights=None)
    in_features = 1408
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"]
    classes = checkpoint.get("classes") or CLASS_NAMES
    img_size = int(checkpoint.get("img_size", 224))
    return state_dict, classes, img_size


def load_model(path):
    state_dict, classes, img_size = load_checkpoint(path)
    model = build_model(len(classes))
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes, img_size


def preprocess_image(image, img_size):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return transform(image).unsqueeze(0).to(DEVICE)


def load_image(path_or_url):
    if str(path_or_url).startswith(("http://", "https://")):
        response = requests.get(path_or_url, timeout=20)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def text_to_bits(text):
    return "".join(format(ord(char), "08b") for char in text)


def bits_to_text(bits, delimiter="<<END>>"):
    chars = []
    for index in range(0, len(bits) - 7, 8):
        chars.append(chr(int("".join(bits[index : index + 8]), 2)))
        if "".join(chars).endswith(delimiter):
            return "".join(chars)[: -len(delimiter)]
    return None


def decode_message_from_bit_iter(bit_iter, delimiter=DELIMITER, max_bytes=4096):
    delimiter_bytes = delimiter.encode("utf-8")
    data = bytearray()
    current = 0
    count = 0

    for bit in bit_iter:
        current = (current << 1) | int(bit)
        count += 1
        if count < 8:
            continue

        data.append(current)
        if len(data) >= len(delimiter_bytes) and data[-len(delimiter_bytes) :] == delimiter_bytes:
            payload = data[:-len(delimiter_bytes)]
            try:
                return payload.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                return None

        if len(data) >= max_bytes:
            return None

        current = 0
        count = 0

    return None


def embed_bits_rgb(cover, message, bits_per_pixel=1):
    if bits_per_pixel < 1 or bits_per_pixel > 4:
        raise ValueError("bits_per_pixel must be between 1 and 4")

    payload = text_to_bits(message + "<<END>>")
    arr = cover.copy()
    flat = arr[:, :, 0].reshape(-1)
    capacity = len(flat) * bits_per_pixel
    if len(payload) > capacity:
        raise ValueError(f"Message too large: need {len(payload)} bits, have {capacity} bits")

    bit_index = 0
    mask_clear = 0xFF ^ ((1 << bits_per_pixel) - 1)
    for pixel_index in range(len(flat)):
        chunk = payload[bit_index : bit_index + bits_per_pixel]
        if not chunk:
            break
        chunk = chunk.ljust(bits_per_pixel, "0")
        flat[pixel_index] = (flat[pixel_index] & mask_clear) | int(chunk, 2)
        bit_index += bits_per_pixel

    arr[:, :, 0] = flat.reshape(arr.shape[:2])
    return arr


def encode_lsb(cover, message):
    return embed_bits_rgb(cover, message, bits_per_pixel=1)


def encode_ssb4(cover, message):
    return embed_bits_rgb(cover, message, bits_per_pixel=4)


def encode_ssbn(cover, message, n=1):
    return embed_bits_rgb(cover, message, bits_per_pixel=max(1, int(n)))


def _fft_positions(shape, window=64):
    height, width = shape
    freq_width = width // 2 + 1
    rows = min(height, window)
    cols = min(freq_width, window)
    positions = []
    for row in range(1, rows):
        for col in range(1, cols):
            positions.append((row, col))
    return positions


def encode_fft(cover, message, window=64, scale=12.0):
    payload = text_to_bits(message + "<<END>>")
    arr = cover.copy().astype(np.float32)
    gray = arr.mean(axis=2)
    spectrum = np.fft.rfft2(gray)
    positions = _fft_positions(gray.shape, window=window)

    if len(payload) > len(positions):
        raise ValueError(
            f"Message too large for FFT embedding: need {len(payload)} coefficients, have {len(positions)}."
        )

    half_scale = scale / 2.0
    for index, bit in enumerate(payload):
        row, col = positions[index]
        coefficient = spectrum[row, col]
        magnitude = np.abs(coefficient)
        phase = np.angle(coefficient)
        quantized = np.round(magnitude / scale) * scale
        target = quantized + (half_scale if bit == "1" else 0.0)
        if target <= 0:
            target = scale if bit == "0" else scale + half_scale
        spectrum[row, col] = target * np.exp(1j * phase)

    stego_gray = np.fft.irfft2(spectrum, s=gray.shape)
    stego_gray = np.clip(stego_gray, 0, 255).astype(np.uint8)
    stego = np.repeat(stego_gray[:, :, None], 3, axis=2)
    return stego


_DCT_MATRIX_CACHE = {}


def _dct_matrix(size):
    matrix = _DCT_MATRIX_CACHE.get(size)
    if matrix is not None:
        return matrix

    matrix = np.zeros((size, size), dtype=np.float32)
    factor = np.pi / size
    scale_0 = np.sqrt(1.0 / size)
    scale = np.sqrt(2.0 / size)
    for k in range(size):
        alpha = scale_0 if k == 0 else scale
        for n in range(size):
            matrix[k, n] = alpha * np.cos((2 * n + 1) * k * factor / 2.0)

    _DCT_MATRIX_CACHE[size] = matrix
    return matrix


def _block_dct(block):
    matrix = _dct_matrix(block.shape[0])
    return matrix @ block @ matrix.T


def _block_idct(coefficients):
    matrix = _dct_matrix(coefficients.shape[0])
    return matrix.T @ coefficients @ matrix


def encode_dct(cover, message, block_size=8, coefficient=(3, 2), step=32.0):
    payload = text_to_bits(message + "<<END>>")
    arr = cover.copy().astype(np.float32)
    gray = arr.mean(axis=2)
    height, width = gray.shape
    blocks_y = height // block_size
    blocks_x = width // block_size
    capacity = blocks_y * blocks_x

    if len(payload) > capacity:
        raise ValueError(
            f"Message too large for DCT embedding: need {len(payload)} blocks, have {capacity}."
        )

    bit_index = 0
    coeff_row, coeff_col = coefficient
    for by in range(blocks_y):
        for bx in range(blocks_x):
            if bit_index >= len(payload):
                break

            y0 = by * block_size
            x0 = bx * block_size
            block = gray[y0 : y0 + block_size, x0 : x0 + block_size]
            dct_block = _block_dct(block)

            bit = payload[bit_index]
            current = float(dct_block[coeff_row, coeff_col])
            sign = 1.0 if current >= 0 else -1.0
            magnitude = abs(current)
            bin_index = int(magnitude // step)
            if (bin_index % 2) != int(bit):
                bin_index += 1
            target = bin_index * step + (step * 0.5)
            dct_block[coeff_row, coeff_col] = sign * target

            gray[y0 : y0 + block_size, x0 : x0 + block_size] = _block_idct(dct_block)
            bit_index += 1

    stego_gray = np.clip(gray, 0, 255).astype(np.uint8)
    stego = np.repeat(stego_gray[:, :, None], 3, axis=2)
    return stego


def decode_bits_rgb(stego, bits_per_pixel=1, max_bytes=4096):
    if bits_per_pixel < 1 or bits_per_pixel > 4:
        raise ValueError("bits_per_pixel must be between 1 and 4")

    flat = stego[:, :, 0].reshape(-1)
    mask = (1 << bits_per_pixel) - 1

    def bit_iter():
        for pixel_value in flat:
            chunk = int(pixel_value) & mask
            for shift in range(bits_per_pixel - 1, -1, -1):
                yield (chunk >> shift) & 1

    return decode_message_from_bit_iter(bit_iter(), max_bytes=max_bytes)


def decode_fft(stego, window=64, scale=12.0, max_bytes=4096):
    gray = stego.astype(np.float32).mean(axis=2)
    spectrum = np.fft.rfft2(gray)
    positions = _fft_positions(gray.shape, window=window)
    half_scale = scale / 2.0

    def bit_iter():
        for row, col in positions:
            magnitude = np.abs(spectrum[row, col])
            yield int(np.round(magnitude / half_scale) % 2)

    return decode_message_from_bit_iter(bit_iter(), max_bytes=max_bytes)


def decode_dct(stego, block_size=8, coefficient=(3, 2), step=32.0, max_bytes=4096):
    gray = stego.astype(np.float32).mean(axis=2)
    height, width = gray.shape
    blocks_y = height // block_size
    blocks_x = width // block_size
    coeff_row, coeff_col = coefficient

    def bit_iter():
        for by in range(blocks_y):
            for bx in range(blocks_x):
                y0 = by * block_size
                x0 = bx * block_size
                block = gray[y0 : y0 + block_size, x0 : x0 + block_size]
                dct_block = _block_dct(block)
                magnitude = abs(float(dct_block[coeff_row, coeff_col]))
                yield int(magnitude // step) % 2

    return decode_message_from_bit_iter(bit_iter(), max_bytes=max_bytes)


def extract_message(image, technique, n=1, max_bytes=4096):
    stego = np.array(image, dtype=np.uint8)
    technique = technique.lower()

    if technique == "lsb":
        return decode_bits_rgb(stego, bits_per_pixel=1, max_bytes=max_bytes)
    if technique == "ssb4":
        return decode_bits_rgb(stego, bits_per_pixel=4, max_bytes=max_bytes)
    if technique == "ssbn":
        return decode_bits_rgb(stego, bits_per_pixel=max(1, int(n)), max_bytes=max_bytes)
    if technique == "dct":
        return decode_dct(stego, max_bytes=max_bytes)
    if technique == "fft":
        return decode_fft(stego, max_bytes=max_bytes)
    return None


def extract_message_auto(image, preferred_technique=None, n=1, max_bytes=4096):
    tried = []
    candidates = []
    if preferred_technique:
        candidates.append(preferred_technique.lower())
    candidates.extend(tech for tech in SUPPORTED_TECHNIQUES if tech not in candidates)

    for technique in candidates:
        tried.append(technique)
        message = extract_message(image, technique, n=n, max_bytes=max_bytes)
        if message:
            return message, technique, tried

    return None, None, tried


def predict_image(model, classes, image, img_size):
    tensor = preprocess_image(image, img_size)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    idx = int(probs.argmax())
    return {
        "label": classes[idx],
        "confidence": float(probs[idx]),
        "all_probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
    }


def classify_and_extract(image_path, binary_bundle, multiclass_bundle, technique_hint=None):
    binary_model, binary_classes, binary_img_size = binary_bundle
    multi_model, multi_classes, multi_img_size = multiclass_bundle
    img_size = binary_img_size if binary_img_size == multi_img_size else min(binary_img_size, multi_img_size)
    image = load_image(image_path)

    binary_result = predict_image(binary_model, binary_classes, image, img_size)
    multi_result = predict_image(multi_model, multi_classes, image, img_size)

    is_stego = binary_result["label"] != "clean"
    extracted_message = None
    decoded_from = None
    preferred = technique_hint or multi_result["label"]
    extracted_message, decoded_from, _ = extract_message_auto(image, preferred, n=1)
    return {
        "is_stego": is_stego,
        "technique": multi_result["label"],
        "decoded_from": decoded_from,
        "confidence": multi_result["confidence"] if is_stego else binary_result["confidence"],
        "message": extracted_message,
        "binary": binary_result,
        "multi": multi_result,
        "all_probs": {
            "binary": binary_result["all_probs"],
            "multi": multi_result["all_probs"],
        },
    }


def attack(args):
    image = load_image(args.image)
    cover = np.array(image.resize((args.size, args.size)), dtype=np.uint8)

    encoders = {
        "lsb": encode_lsb,
        "ssb4": encode_ssb4,
        "ssbn": lambda c, m: encode_ssbn(c, m, n=args.n),
        "dct": encode_dct,
        "fft": encode_fft,
    }
    stego = encoders[args.technique](cover, args.message)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(stego).save(output_path)
    print(f"[OK] '{args.message}' hidden via {args.technique.upper()} -> {output_path}")


def detect(args):
    binary_model = load_model(BINARY_MODEL_PATH)
    multi_model = load_model(MULTICLASS_MODEL_PATH)
    result = classify_and_extract(
        args.image,
        binary_model,
        multi_model,
        technique_hint=args.technique,
    )

    binary = result["binary"]
    multi = result["multi"]
    print(f"[{ 'STEGO' if result['is_stego'] else 'CLEAN' }] Binary: {binary['label'].upper()} ({binary['confidence']:.1%})")
    print(f"  Multi-class: {multi['label'].upper()} ({multi['confidence']:.1%})")
    if result["decoded_from"] and result["decoded_from"] != result["technique"]:
        print(f"  Decoded from: {result['decoded_from'].upper()} (model guess was {result['technique'].upper()})")
    print(f"  Extracted message: {result['message'] or '(none)'}")
    print(f"  Binary probs: {binary['all_probs']}")
    print(f"  Multi probs : {multi['all_probs']}")


def evaluate(args):
    binary_model = load_model(BINARY_MODEL_PATH)
    multi_model = load_model(MULTICLASS_MODEL_PATH)
    binary_bundle = binary_model
    multi_bundle = multi_model

    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.dir:
        image_paths.extend(
            path for path in sorted(Path(args.dir).iterdir())
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        )

    if not image_paths:
        raise ValueError("Provide --image or --dir for evaluation")

    binary_counts = Counter()
    multi_counts = Counter()
    for path in image_paths:
        result = classify_and_extract(path, binary_bundle, multi_bundle, technique_hint=args.technique)
        binary_counts[result["binary"]["label"]] += 1
        multi_counts[result["multi"]["label"]] += 1
        print(
            f"{path.name}: binary={result['binary']['label']} ({result['binary']['confidence']:.1%}), "
            f"multi={result['multi']['label']} ({result['multi']['confidence']:.1%}), "
            f"decoded_from={result['decoded_from'] or '(none)'}, "
            f"message={result['message'] or '(none)'}"
        )

    print("\nSummary")
    print(f"  Binary classes : {dict(binary_counts)}")
    print(f"  Multi classes  : {dict(multi_counts)}")


def main():
    parser = argparse.ArgumentParser(description="Local Stego Attack and Detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_attack = subparsers.add_parser("attack", help="Hide a message inside an image")
    p_attack.add_argument("--image", required=True, help="Input cover image")
    p_attack.add_argument("--message", required=True, help="Message to hide")
    p_attack.add_argument("--technique", default="lsb", choices=["lsb", "ssb4", "ssbn", "dct", "fft"])
    p_attack.add_argument("--n", type=int, default=1, help="Bit depth for ssbn")
    p_attack.add_argument("--size", type=int, default=224, help="Resize before embedding")
    p_attack.add_argument("--output", default=str(DEFAULT_RESULTS_DIR / "stego_output.png"))
    p_attack.set_defaults(func=attack)

    p_detect = subparsers.add_parser("detect", help="Run both models on one image")
    p_detect.add_argument("--image", required=True, help="Image to classify")
    p_detect.add_argument("--technique", choices=SUPPORTED_TECHNIQUES, help="Hint the embedded technique to try first")
    p_detect.set_defaults(func=detect)

    p_eval = subparsers.add_parser("evaluate", help="Evaluate one image or a folder with both models")
    p_eval.add_argument("--image", help="Single image to evaluate")
    p_eval.add_argument("--dir", help="Folder of images to evaluate")
    p_eval.add_argument("--technique", choices=SUPPORTED_TECHNIQUES, help="Hint the embedded technique to try first")
    p_eval.set_defaults(func=evaluate)

    args = parser.parse_args()
    if not BINARY_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {BINARY_MODEL_PATH}")
    if not MULTICLASS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MULTICLASS_MODEL_PATH}")
    args.func(args)


if __name__ == "__main__":
    main()
