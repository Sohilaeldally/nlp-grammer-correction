# English Grammar Correction 📝
<img width="1920" height="1080" alt="Screenshot (5809)" src="https://github.com/user-attachments/assets/e9c931fb-4cb6-4e56-9d09-11d366d569ac" />

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![Transformers](https://img.shields.io/badge/Transformers-T5-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)

A deep learning project to **automatically correct English grammar** using a **T5 transformer model**.  
Includes training on a custom dataset, evaluation, and a **Streamlit web app** for real-time grammar correction.

---

## 📂 Project Structure

project-root/
│
├── data/ # Dataset folder
│ └── Grammer Correction.csv
│
├── models/ # Trained models will be saved here
│ └── best_model/
│
├── notebooks/
│ ├── data_exploration.ipynb # Exploratory Data Analysis
│ └── colab_run.ipynb # Training notebook (for Colab)
│
├── src/
│ ├── app.py # Streamlit app for grammar correction
│ ├── train.py # Training script
│ ├── dataset.py # Custom PyTorch dataset
│ └── metrics.py # Evaluation metrics (ROUGE, Edit Distance)
│
├── requirements.txt # Required Python packages
└── README.md


---

## ⚙️ Installation

1. Clone this repository:

```bash
git clone <your-repo-url>
cd project-root
