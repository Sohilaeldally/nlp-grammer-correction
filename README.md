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
It contains two main columns:
- **Ungrammatical Statement** → Input sentences  
- **Standard English** → Corrected sentences (labels)
  
Preprocessing and tokenization for T5 are handled in `dataset.py.` 

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

