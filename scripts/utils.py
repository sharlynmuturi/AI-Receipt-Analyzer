import pytesseract
from PIL import Image
from fuzzywuzzy import fuzz
import json

def extract_words_boxes(image: Image.Image, conf_threshold: int = 40):
    """
    Extract OCR words and bounding boxes from an image.
    Returns words and normalized boxes (0–1000).
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    w, h = image.size

    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        try:
            conf = int(data["conf"][i])
        except ValueError:
            continue
        if conf < conf_threshold:
            continue

        x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        box = [int(1000 * x / w), int(1000 * y / h), int(1000 * (x + bw) / w), int(1000 * (y + bh) / h)]
        words.append(text)
        boxes.append(box)
    return words, boxes

def fuzzy_match(a: str, b: str, threshold: int = 85):
    return fuzz.ratio(a.lower(), b.lower()) >= threshold

def assign_labels(words: list, ground_truth: dict):
    """
    Assign BIO labels to OCR words based on fuzzy matching to ground-truth values.
    """
    labels = ["O"] * len(words)
    texts = [w for w in words]

    for field, value in ground_truth.items():
        if not value:
            continue
        value_tokens = value.split()
        n = len(value_tokens)

        for i in range(len(texts) - n + 1):
            window = texts[i:i+n]
            window_text = " ".join(window)
            if fuzzy_match(window_text, value):
                labels[i] = f"B-{field.upper()}"
                for j in range(1, n):
                    labels[i+j] = f"I-{field.upper()}"
    return labels

def load_ground_truth(label_file_path):
    with open(label_file_path) as f:
        return json.load(f)