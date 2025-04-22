from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import pandas as pd

app = Flask(__name__)

# Загрузка моделей
retriever = SentenceTransformer('models/sentence_model')
generator = T5ForConditionalGeneration.from_pretrained('models/rut5_model')
tokenizer = T5Tokenizer.from_pretrained('models/rut5_model')

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