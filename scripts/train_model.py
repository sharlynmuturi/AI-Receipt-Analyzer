# train LayoutLMv3
from pathlib import Path
import numpy as np
from transformers import LayoutLMv3ForTokenClassification, Trainer, TrainingArguments, LayoutLMv3Processor
from datasets import load_from_disk
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).parent
ARTIFACTS = BASE_DIR / "artifacts"
TRAIN_DATASET = ARTIFACTS / "encoded_train_receipts_dataset"
TEST_DATASET = ARTIFACTS / "encoded_test_receipts_dataset"
MODEL_PATH = ARTIFACTS / "receipt_model"

# Label mapping
label_list = ["O","B-COMPANY","I-COMPANY","B-DATE","I-DATE","B-TOTAL","I-TOTAL","B-ADDRESS","I-ADDRESS"]
label2id = {l:i for i,l in enumerate(label_list)}
id2label = {i:l for l,i in label2id.items()}

train_dataset = load_from_disk(TRAIN_DATASET)
test_dataset = load_from_disk(TEST_DATASET)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    labels = p.label_ids
    true_preds, true_labels = [], []
    for pred, lab in zip(preds, labels):
        for p_i, l_i in zip(pred, lab):
            if l_i != -100:
                true_preds.append(id2label[p_i])
                true_labels.append(id2label[l_i])
    report = classification_report(true_labels, true_preds, output_dict=True, zero_division=0)
    return {"precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"]}

training_args = TrainingArguments(
    output_dir="./layoutlmv3_receipts",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=100,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)

trainer.save_model(MODEL_PATH)