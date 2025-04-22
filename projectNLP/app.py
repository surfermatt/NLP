from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import pandas as pd

import os
import requests
import zipfile

def download_and_unpack_model():
    model_url = "https://vh297-fm.sweb.ru/files/surfermatt_ru/ML/projectNLP/models/rut5_model"
    model_dir = "models/rut5_model"

    if not os.path.exists(model_dir):
        print("📥 Загружаем модель...")
        r = requests.get(model_url)
        with open("rut5_model.zip", "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile("rut5_model.zip", 'r') as zip_ref:
            zip_ref.extractall("models")
        os.remove("rut5_model.zip")
        print("✅ Модель загружена и распакована.")

download_and_unpack_model()

app = Flask(__name__)

download_and_unpack_model()
tokenizer = T5Tokenizer.from_pretrained("models/rut5_model")
generator = T5ForConditionalGeneration.from_pretrained("models/rut5_model")

# Загрузка моделей
retriever = SentenceTransformer('models/sentence_model')
#generator = T5ForConditionalGeneration.from_pretrained('models/rut5_model')
#tokenizer = T5Tokenizer.from_pretrained('models/rut5_model')

# Загрузка текстов и эмбеддингов
corpus = pd.read_csv('models/journal_texts.csv')['description'].tolist()
corpus_embeddings = np.load('models/journal_embeddings.npy')
corpus_embeddings = torch.tensor(corpus_embeddings)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query = data.get('description', '')

    if not query:
        return jsonify({'error': 'No description provided'}), 400

    # Семантический поиск
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)[0]
    best_match_idx = hits[0]['corpus_id']
    best_match_text = corpus[best_match_idx]

    # Генерация рекомендации
    input_text = f'неисправность: {best_match_text} рекомендация:'
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output_ids = generator.generate(input_ids, max_length=100)
    recommendation = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({
        'matched_issue': best_match_text,
        'recommendation': recommendation
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
