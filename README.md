
## 🚀 Chatbot

This project builds a **custom chatbot** that uses a neural network (** lstm**) to generate embeddings for questions and utilizes a **FAISS (Facebook AI Similarity Search)** index to match user queries with relevant answers. Additionally, the chatbot supports **voice input and output** for a seamless user experience.

---

## 📚 Project Structure

```
/futuristic-faq-chatbot
├── /model
│   ├── chatbot_model.h5           # Trained chatbot model
│   ├── faq_index.faiss            # FAISS index for question embeddings
│   └── answer_lookup.npy          # Lookup array for answers
├── /static
│   ├── css/style.css              # Custom CSS for frontend
│   └── js/voice.js                # JavaScript for voice recognition and speech synthesis
├── /templates
│   └── index.html                 # Chatbot UI
├── app.py                         # Flask server and chat API
├── train.py                       # Model training and FAISS indexing
└── faq_dataset.csv   # FAQ dataset
```

---

## ⚡️ How It Works

### 1. Preprocessing
- Load FAQ data from `high_quality_faq_dataset.csv`.
- Tokenize and convert questions into padded sequences.
- Create a lookup array to map questions to their respective answers.

### 2. Model Training
- Define a sequential **LSTM model** using Keras with the following layers:
    - Embedding Layer
    - LSTM Layer
    - GlobalMaxPooling Layer
    - Dense Layer for classification
- Train the model on tokenized question sequences.

### 3. FAISS Index Creation
- Use the trained model to generate embeddings for the FAQ dataset.
- Store these embeddings in a FAISS index for fast similarity search.
- Save the model, FAISS index, and answers lookup for future use.

---

## 🎧 Voice Feature

### 🔦 Voice Input
- Users can speak their questions directly using the **Speak** button.
- JavaScript captures voice input using the `Web Speech API` and sends the recognized text to the chatbot.

### 🔊 Voice Output
- The chatbot responds with voice output using `speechSynthesis` after returning the answer.

---

## 🧬 Training the Model

Run `train.py` to train the model and generate the FAISS index.
```bash
python train.py
```
👌 The model, FAISS index, and answer lookup will be saved in the `/model` directory.

---

## 🖥️ Running the Application

1. **Run Flask Application**
```bash
python app.py
```

2. **Access the Chatbot UI**
- Open your browser and visit:
```
http://127.0.0.1:5000
```

---

## 🔥 Usage

- Ask a question by typing or using the **Speak** button.
- The chatbot processes your query and responds with the best match.
- If no match is found, the chatbot returns a default error message.
- Voice output is automatically generated for each response.

---

## 📝 API Endpoint

### POST `/chat`
- **Description:** Handles user queries and returns the best-matching FAQ answer.
- **Request Body:**
```json
{
  "query": "What is cloud computing?"
}
```
- **Response:**
```json
{
  "response": "Cloud computing is the delivery of computing services over the internet."
}
```

---

## 🎧 Voice Integration

### 🔥 Enabling Voice Input and Output
- JavaScript (`/static/js/voice.js`) adds support for:
    - **Speech Recognition:** Converts user voice to text.
    - **Speech Synthesis:** Converts chatbot responses to voice.
- Voice features can be enabled or modified by editing `index.html` and `voice.js`.

---

![Screenshot 2025-03-20 161636](https://github.com/user-attachments/assets/4be1df5d-dc34-4d0c-8049-5fd8e9f9352a)

