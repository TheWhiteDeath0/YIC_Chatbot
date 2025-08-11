# chatbot/nlp_processor.py

import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Intent

# --- This section remains the same ---
def load_chat_data():
    all_patterns = []
    pattern_to_tag = {}
    tag_map = {}

    for intent in Intent.objects.prefetch_related('patterns', 'responses'):
        tag_map[intent.tag] = [r.text for r in intent.responses.all()]
        for p in intent.patterns.all(): 
            all_patterns.append(p.text)
            pattern_to_tag[p.text] = intent.tag

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(all_patterns)
    return vectorizer, X, all_patterns, pattern_to_tag, tag_map

try:
    vectorizer, X_patterns, all_patterns, pattern_to_tag, tag_map = load_chat_data()
except Exception as e:
    print("[Chatbot Load Error]", e)
    vectorizer, X_patterns, all_patterns, pattern_to_tag, tag_map = None, None, [], {}, {}
# -----------------------------------------


# --- UPDATED get_response function ---
def get_response(user_input):
    start_time = time.time()
    
    # *** 1. RULE-BASED LOGIC STARTS HERE ***
    # We define simple rules for common words.
    # The key is the user's input (in lowercase), and the value is the intent tag.
    
    lowered_input = user_input.lower()
    
    # Rule for greetings
    if any(word in lowered_input for word in ["hello", "hi", "hey"]):
        matched_tag = "greeting"
        response_text = random.choice(tag_map[matched_tag])
        confidence = 1.0 # Rules are 100% confident
        method = "Rule-based"

    # Rule for goodbyes
    elif any(word in lowered_input for word in ["bye", "goodbye", "thanks"]):
        matched_tag = "goodbye"
        response_text = random.choice(tag_map[matched_tag])
        confidence = 1.0
        method = "Rule-based"

    # *** 2. IF NO RULE MATCHES, FALL BACK TO RETRIEVAL-BASED LOGIC ***
    else:
        method = "Retrieval" # Set the method to retrieval
        user_vector = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, X_patterns)
        most_similar_idx = similarities.argmax()
        confidence = similarities[0, most_similar_idx]
        
        similarity_threshold = 0.5

        if confidence >= similarity_threshold:
            matched_pattern = all_patterns[most_similar_idx]
            matched_tag = pattern_to_tag[matched_pattern]
            response_text = random.choice(tag_map[matched_tag])
        else:
            # Fallback if retrieval confidence is also too low
            matched_tag = "fallback"
            confidence = 0.0
            fallback_responses = tag_map.get("fallback", ["I'm sorry, I don't understand that."])
            response_text = random.choice(fallback_responses)

    # *** 3. FINALIZE AND RETURN THE DATA ***
    end_time = time.time()
    processing_time = round((end_time - start_time) * 1000)

    response_data = {
        "response": response_text,
        "method": method,
        "tag": matched_tag,
        "confidence": confidence,
        "processing_time": processing_time
    }

    return response_data