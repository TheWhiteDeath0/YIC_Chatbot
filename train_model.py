# train_model.py

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

# --- 1. Load the data from intents.json ---
print("Loading data from intents.json...")
with open('intents.json', 'r') as f:
    data = json.load(f)

# --- 2. Prepare the data for training ---
training_sentences = []
training_labels = []
labels = []
responses = {}

for intent in data['intents']:
    # Add all patterns to the training sentences
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    
    # Store the responses for later use
    responses[intent['tag']] = intent['responses']
    
    # Add the tag to the labels list if not already there
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

num_classes = len(labels)

# --- 3. Encode the labels ---
print("Encoding labels...")
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

# --- 4. Tokenize and pad the sentences ---
print("Tokenizing and padding sentences...")
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# --- 5. Build the LSTM Model ---
print("Building the LSTM model...")
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

model.summary()

# --- 6. Train the Model ---
print("Training the model...")
epochs = 100
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

# --- 7. Save the Trained Model and other necessary objects ---
print("Saving the model and other necessary files...")

# Save the trained model
model.save("chatbot_model.h5")

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Save the label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Model training complete and all files saved!")