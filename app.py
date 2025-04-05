from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle

with open('model1.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
model.eval()

LABELS = {0: "real", 1: "false"}

@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in the 'templates' folder

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
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
