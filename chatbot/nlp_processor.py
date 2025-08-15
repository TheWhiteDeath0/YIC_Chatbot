# chatbot/nlp_processor.py

import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
# We need to import the TF-IDF components again
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Load all necessary models and files ---
print("Loading all chatbot models and supporting files...")
try:
    # --- LSTM Model Components ---
    lstm_model = tf.keras.models.load_model('chatbot_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
        
    # --- Intents Data ---
    with open('intents.json', 'r') as f:
        data = json.load(f)

    # --- TF-IDF Retrieval Model Components ---
    all_patterns = []
    pattern_to_tag_retrieval = {}
    for intent in data['intents']:
        for pattern in intent['patterns']:
            all_patterns.append(pattern)
            pattern_to_tag_retrieval[pattern] = intent['tag']
            
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_patterns_retrieval = tfidf_vectorizer.fit_transform(all_patterns)

except Exception as e:
    print(f"[Chatbot Load Error] {e}")
    # Set all models to None if there's an error
    lstm_model, tokenizer, lbl_encoder, data, tfidf_vectorizer, X_patterns_retrieval, all_patterns, pattern_to_tag_retrieval = [None]*8


# --- 2. The Final 3-Stage Hybrid get_response function ---
def get_response(user_input):
    start_time = time.time()
    if not all([lstm_model, data, tfidf_vectorizer]):
        return { "response": "Error: The chatbot models are not loaded correctly. Please check the server logs." }

    lowered_input = user_input.lower()
    method, predicted_tag, confidence = "Unknown", "fallback", 0.0

    # --- STAGE 1: RULE-BASED CHECK ---
    # Define a simple function to reduce repetition
    def set_rule_based_tag(tag):
        return "Rule-based", tag, 1.0

    if any(word in lowered_input for word in ["hello", "hi", "hey"]):
        method, predicted_tag, confidence = set_rule_based_tag("greeting")
    elif any(word in lowered_input for word in ["bye", "goodbye", "thanks", "thank you"]):
        method, predicted_tag, confidence = set_rule_based_tag("goodbye")
    elif "hnd" in lowered_input.split():
        method, predicted_tag, confidence = set_rule_based_tag("about_hnd")
    elif "igcse" in lowered_input.split():
        method, predicted_tag, confidence = set_rule_based_tag("about_igcse")
    elif "ged" in lowered_input.split():
        method, predicted_tag, confidence = set_rule_based_tag("about_ged")
    
    # --- STAGE 2: TF-IDF RETRIEVAL CHECK ---
    if method == "Unknown":
        user_vector = tfidf_vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, X_patterns_retrieval)
        retrieval_confidence = similarities.max()
        
        # Set a threshold for the retrieval model
        retrieval_threshold = 0.7 
        if retrieval_confidence > retrieval_threshold:
            method = "Retrieval"
            confidence = float(retrieval_confidence)
            most_similar_idx = similarities.argmax()
            matched_pattern = all_patterns[most_similar_idx]
            predicted_tag = pattern_to_tag_retrieval[matched_pattern]

    # --- STAGE 3: LSTM PREDICTION ---
    if method == "Unknown":
        method = "LSTM" # Final attempt with LSTM
        max_len = 20
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, truncating='post', maxlen=max_len)
        result = lstm_model.predict(padded_sequence, verbose=0)[0]
        
        confidence = float(result.max())
        
        # Set a threshold for the LSTM model
        lstm_threshold = 0.75
        if confidence >= lstm_threshold:
            predicted_tag_index = np.argmax(result)
            predicted_tag = lbl_encoder.inverse_transform([predicted_tag_index])[0]
        else:
            predicted_tag = "fallback"

    # --- FINALIZE AND GET RESPONSE TEXT ---
    response_text = "I'm sorry, I don't understand that. Please rephrase." # Default fallback
    for i in data['intents']:
        if i['tag'] == predicted_tag:
            response_text = np.random.choice(i['responses'])
            break

    end_time = time.time()
    processing_time = round((end_time - start_time) * 1000)

    # Prepare final data dictionary
    response_data = {
        "response": response_text,
        "method": method,
        "tag": predicted_tag,
        "confidence": confidence,
        "processing_time": processing_time
    }
    return response_data