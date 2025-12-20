# English Grammar Correction 📝

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![Transformers](https://img.shields.io/badge/Transformers-T5-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)

A deep learning project to **automatically correct English grammar** using a **T5 transformer model**.  
Includes training on a custom dataset, evaluation, and a **Streamlit web app** for real-time grammar correction.


<img width="1920" height="1080" alt="Screenshot (5809)" src="https://github.com/user-attachments/assets/e9c931fb-4cb6-4e56-9d09-11d366d569ac" />

---

## 📂 Project Structure

```text
project-root/
│
├── data/
│   └── Grammer Correction.csv
│
├── models/
│   └── best_model/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── colab_run.ipynb
│
├── src/
│   ├── app.py
│   ├── train.py
│   ├── dataset.py
│   └── metrics.py
│
├── requirements.txt
└── README.md


```
## ⚙️ Installation

#### 1. Clone the repository
```bash
git clone <your-repo-url>
cd project-root
```
#### 2. Install dependencies
```bash
pip install -r requirements.txt
```
**Main libraries used:** `torch` `transformers` `pandas` `matplotlib` `seaborn` `evaluate` `Levenshtein` `tqdm` `streamlit`

## 📝 Dataset
The dataset is located at:
```text
data/Grammar Correction.csv
```
It originally contains **three columns**:
- **Ungrammatical Statement** → Input sentences  
- **Standard English** → Corrected sentences (labels)
- **Error Type** → Type of grammatical error (e.g., Verb Tense Errors, Subject-Verb Agreement, Article Usage)


⚠️ **Note:**  
The **Error Type** column is **not used during training** and is dropped during preprocessing.  
The model is trained only on sentence-level correction without explicit error-type supervision.

Preprocessing and tokenization for T5 are handled in `dataset.py.` 

## 🧪 Testing / Inference (Before Fine-Tuning)

You can try the pre-trained **T5 grammar correction model** before fine-tuning it on your dataset.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model
model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction").to(device)
tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")

sentences = [
    "Him and me was going to the market yesterday.",
    "I has a pen."
]

inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)

with torch.no_grad():
    outputs = model.generate(**inputs)

preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for src, pred in zip(sentences, preds):
    print(f"Input: {src}")
    print(f"Prediction: {pred}")```

```python
Input: Him and me was going to the market yesterday.
Prediction:  Him and me were going to the market yesterday.```

Input: I has a pen.
Prediction: I have a pen.

✅ This shows how the base model performs before fine-tuning on your dataset. After fine-tuning (train.py), the model will better match your dataset’s style and grammar patterns. how to writhe this in readme

## 🚀 Training

To train the model, run:
```bash
python src/train.py
```

### Training details

- Uses **T5-base grammar correction model** `(vennify/t5-base-grammar-correction)` as initialization
- Dataset split:
  - 80% training
  - 20% validation
- Trains for **25 epochs**
- **Early stopping** with patience = 4
- Saves the **best model** (based on ROUGE-L) to:

```text
models/best_model/
```
### Evaluation metrics
`ROUGE-1`, `ROUGE-2`, `ROUGE-L`, `Normalized Edit Distance (Levenshtein)`

## 🌐 Streamlit Web App
Run the interactive Streamlit application using:
```bash
streamlit run src/app.py
```
### How it works

1. Enter a sentence with grammatical errors
2. Click **Correct**
3. The corrected sentence is displayed instantly

## 📊 Results
### Sample Prediction

| Input Sentence                               | Corrected Sentence                          |
|----------------------------------------------|--------------------------------------------|
| He go to school yesterday.                   | He went to school yesterday.               |
| I has a pen.                                 | I have a pen.                              |
|She don’t knows nothing about the project yet.| She doesn’t know anything about the project yet. |

### Validation Metrics 

- **ROUGE-1**: `0.9508`
- **ROUGE-2**: `0.8957`
- **ROUGE-L**: `0.9422`
- **Normalized Edit Distance**:`0.0611`
## 🔧 Notes

- The model supports **CPU** and **GPU**
- For faster training, use Google Colab:
```text
notebooks/colab_run.ipynb
```
- Data exploration is available in:
```text
notebooks/data_exploration.ipynb
```
- You can adjust `max_length` and `batch_size` in `train.py` depending on available GPU memory

## ⭐ Future Improvements

- Fine-tune on a **larger dataset**
- Add **multi-lingual grammar correction**
- Enhance Streamlit UI with **batch sentence correction**
- Deploy the app online (e.g., **Hugging Face Spaces** or **Render**)
