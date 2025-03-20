import pandas as pd
import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load FAQ Data from CSV
data_path = 'faq_dataset.csv'
faq_data = pd.read_csv(data_path)

# Extract Questions and Answers
answers = faq_data['answer'].tolist()

# Load Model and FAISS Index
model = load_model('model/chatbot_model.h5')
index = faiss.read_index('model/faq_index.faiss')

# Tokenizer Configuration
MAX_VOCAB_SIZE = 1000
MAX_SEQ_LEN = 20

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(faq_data['question'].tolist())

# Preprocess Input Query
def preprocess_query(query):
    seq = tokenizer.texts_to_sequences([query])
    padded_seq = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding='post')
    return padded_seq

# Find Best Match Using FAISS
def find_best_match(query_embedding):
    query_embedding = np.array(query_embedding).reshape(1, -1)
    D, I = index.search(query_embedding, 1)  # Get top 1 match
    return I[0][0], D[0][0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """API Endpoint to Handle Chat Queries"""
    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Preprocess Query and Get Embeddings
    padded_seq = preprocess_query(query)
    query_embedding = model.predict(padded_seq)

    # Find Best Match
    match_idx, distance = find_best_match(query_embedding)

    if distance > 5.0:  # Set threshold for similarity
        response = "Sorry, I couldn't find a relevant answer."
    else:
        response = answers[match_idx]

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

