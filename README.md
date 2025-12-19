# English Grammar Correction 📝

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![Transformers](https://img.shields.io/badge/Transformers-T5-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)

A deep learning project to **automatically correct English grammar** using a **T5 transformer model**.  
Includes training on a custom dataset, evaluation, and a **Streamlit web app** for real-time grammar correction.
<img width="1920" height="1080" alt="Screenshot (5809)" src="https://github.com/user-attachments/assets/e9c931fb-4cb6-4e56-9d09-11d366d569ac" />

⚙️ Installation

Clone this repository:

git clone <your-repo-url>
cd project-root


Install required packages:

pip install -r requirements.txt


Requirements include: torch, transformers, pandas, matplotlib, seaborn, evaluate, Levenshtein, tqdm, streamlit

📝 Dataset

The dataset is stored in data/Grammer Correction.csv

It contains:

Ungrammatical Statement → Input sentences

Standard English → Corrected sentences (labels)

Preprocessing is done in dataset.py using tokenization for T5.

🚀 Training

Training is done using train.py:

python src/train.py


Uses T5-base grammar correction model (vennify/t5-base-grammar-correction) as a starting point

Splits the dataset into 80% training / 20% validation

Trains for 25 epochs with early stopping (patience = 4)

Saves the best model (based on ROUGE-L) in models/best_model/

Metrics computed:

ROUGE-1, ROUGE-2, ROUGE-L

Normalized Edit Distance (Levenshtein)

🧪 Testing / Inference

Test the model with custom sentences:

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSeq2SeqLM.from_pretrained("models/best_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("models/best_model")

sentences = [
    "He go to school yesterday.",
    "I has a pen."
]

inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)

with torch.no_grad():
    outputs = model.generate(**inputs)

preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for src, pred in zip(sentences, preds):
    print(f"Input: {src}")
    print(f"Prediction: {pred}")

🌐 Streamlit Web App

Run the interactive web app with:

streamlit run src/app.py


Enter a sentence and click Correct.

The app will display the corrected sentence in real-time.

📊 Results

Sample predictions:

Input Sentence	Corrected Sentence
He go to school yesterday.	He went to school yesterday.
I has a pen.	I have a pen.
She don’t knows nothing about the project yet.	She doesn’t know anything about the project yet.

Metrics (on validation set):

ROUGE-1: 0.92

ROUGE-2: 0.87

ROUGE-L: 0.91

Normalized Edit Distance: 0.08

🔧 Notes

The model can run on CPU or GPU

Use Colab notebooks (notebooks/colab_run.ipynb) if you want GPU acceleration for training

Data exploration and preprocessing is in notebooks/data_exploration.ipynb

Adjust max_length and batch_size in train.py depending on your GPU memory

⭐ Future Improvements

Fine-tune with a larger dataset for better coverage

Add multi-lingual support

Improve UI in Streamlit with batch sentence correction

Deploy the web app online (e.g., Streamlit Cloud)

