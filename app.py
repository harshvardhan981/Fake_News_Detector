from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

MODEL_PATH = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

LABELS = {0: "real", 1: "false"}

@app.route('/')
def home():
    return render('index.html')

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", trunctation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return LABELS[predicted_class]

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json
    text = data.get('text', "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    prediction = predict(text)
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)