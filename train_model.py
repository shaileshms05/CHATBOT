import pandas as pd
import numpy as np
import faiss
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Dataset from CSV
data_path = 'faq_dataset.csv'
faq_data = pd.read_csv(data_path)

# Extract Questions and Answers
questions = faq_data['question'].tolist()
answers = faq_data['answer'].tolist()

# Tokenization and Preprocessing
MAX_VOCAB_SIZE = 1000
MAX_SEQ_LEN = 20

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(questions)
word_index = tokenizer.word_index

# Convert Text to Sequences
sequences = tokenizer.texts_to_sequences(questions)
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

# Prepare Training Data
X = padded_sequences
y = np.array(range(len(answers)))  # Labels as indexes for answer lookup

# Split Data for Training and Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Neural Network Model
EMBEDDING_DIM = 50

model = Sequential([
    Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQ_LEN),
    LSTM(64, return_sequences=True),
    GlobalMaxPooling1D(),
    Dense(32, activation='relu'),
    Dense(len(answers), activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train Model
model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=8)

# Save Model
model.save('model/chatbot_model.h5')

# Generate Embeddings for FAQ
faq_embeddings = model.predict(padded_sequences)

# Store Embeddings in FAISS
d = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(faq_embeddings)

# Save FAISS Index and Answers Lookup
faiss.write_index(index, 'model/faq_index.faiss')
np.save('model/answer_lookup.npy', np.array(answers, dtype=object))

print("✅ Model and FAISS index saved successfully!")
