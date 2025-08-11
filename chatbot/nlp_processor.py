# chatbot/nlp_processor.py

import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# --- Import our new pattern generator ---
from pattern_generator import generate_patterns, save_new_patterns

# --- Load all necessary models and files ---
print("Loading all chatbot models and supporting files...")
try:
    # --- LSTM Model Components ---
    lstm_model = tf.keras.models.load_model('chatbot_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    with open('intents.json', 'r') as f:
        data = json.load(f)
except Exception as e:
    print("[Chatbot Load Error]", e)
    lstm_model, tokenizer, lbl_encoder, data = None, None, None, None


# --- The Final Hybrid get_response function ---
def get_response(user_input):
    start_time = time.time()
    if not all([lstm_model, tokenizer, lbl_encoder, data]):
        return { "response": "Error: The chatbot model is not loaded correctly." }

    # --- LSTM Prediction Logic (This is the primary method now) ---
    max_len = 20
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, truncating='post', maxlen=max_len)
    result = lstm_model.predict(padded_sequence, verbose=0)[0]
    
    predicted_tag_index = np.argmax(result)
    confidence = float(result[predicted_tag_index])
    
    predicted_tag = "fallback" # Default tag
    confidence_threshold = 0.75

    if confidence >= confidence_threshold:
        predicted_tag = lbl_encoder.inverse_transform([predicted_tag_index])[0]

        # --- SELF-IMPROVEMENT STEP ---
        # If confidence is very high, let's generate and save new patterns
        if confidence > 0.90: 
            new_patterns = generate_patterns(user_input)
            save_new_patterns(predicted_tag, user_input, new_patterns)
            
    # --- Get the response text ---
    response_text = "I'm sorry, I don't understand that. Please rephrase."
    for i in data['intents']:
        if i['tag'] == predicted_tag:
            response_text = np.random.choice(i['responses'])
            break

    end_time = time.time()
    processing_time = round((end_time - start_time) * 1000)

    # Prepare final response data
    response_data = {
        "response": response_text,
        "method": "LSTM",
        "tag": predicted_tag,
        "confidence": confidence,
        "processing_time": processing_time
    }
    return response_data