from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction").to(device)
tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")


sentences = [
    "He go to school yesterday.",
    "I has a pen.",
    "He suggested me to go to the doctor because I am sick.",
    "Despite of being tired, but she continued working.",
    "I have visited Paris last year for the first time.",
    "Everyone should knows their responsibilities",
    "She don’t knows nothing about the project yet.",
    "Him and me was going to the market yesterday.",
    "If I would have seen her, I would tell her the truth.",
    "Running fastly, the race was won by him.",
    "The informations you gave me are very helpful.",
    "I am agree with you about this idea."
]

inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)

with torch.no_grad():
    outputs = model.generate(**inputs)

preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

counter=1
for src, pred in zip(sentences, preds):
    print(f"Input {counter}: {src}")
    print(f"Prediction: {pred}")
    print("----------")
    counter+=1
