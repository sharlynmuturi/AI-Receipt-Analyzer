# encodes receipts into torch datasets
from pathlib import Path
from PIL import Image
from transformers import LayoutLMv3Processor
from datasets import Dataset
from utils import extract_words_boxes, assign_labels, load_ground_truth
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "receipts" / "train"
TEST_PATH = BASE_DIR / "receipts" / "test"
MODEL_PATH = BASE_DIR / "artifacts" / "receipt_model"

processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH)

# Label mapping
label_list = ["O","B-COMPANY","I-COMPANY","B-DATE","I-DATE","B-TOTAL","I-TOTAL","B-ADDRESS","I-ADDRESS"]
label2id = {l:i for i,l in enumerate(label_list)}
id2label = {i:l for l,i in label2id.items()}

def encode_dataset(img_folder, label_folder):
    images = list(img_folder.glob("*.jpg"))
    dataset_list = []

    for img_path in tqdm(images):
        image = Image.open(img_path).convert("RGB")
        words, boxes = extract_words_boxes(image)
        gt = load_ground_truth(label_folder / f"{img_path.stem}.txt")
        labels = assign_labels(words, gt)

        encoding = processor(
            image,
            words,
            boxes=boxes,
            word_labels=[label2id[l] for l in labels],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        dataset_list.append({k:v.squeeze() for k,v in encoding.items()})

    dataset = Dataset.from_list(dataset_list)
    dataset.set_format(type="torch")
    return dataset

# Encode and save
train_dataset = encode_dataset(TRAIN_PATH/"img", TRAIN_PATH/"entities")
test_dataset = encode_dataset(TEST_PATH/"img", TEST_PATH/"entities")

train_dataset.save_to_disk(BASE_DIR / "artifacts" / "encoded_train_receipts_dataset")
test_dataset.save_to_disk(BASE_DIR / "artifacts" / "encoded_test_receipts_dataset")