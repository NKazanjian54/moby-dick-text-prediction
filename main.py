import spacy
import re
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def clean_and_tokenize(text, nlp):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    return tokens

def create_sequences(tokens, sequence_length=25):
    sequences = []
    for i in range(len(tokens) - sequence_length):
        seq = tokens[i:i + sequence_length + 1]
        sequences.append(' '.join(seq))
    return sequences

nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
nlp.max_length = 1500000

file_path = 'moby_dick_full.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

tokens = clean_and_tokenize(text, nlp)
sequences = create_sequences(tokens, sequence_length=50)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences)
sequences_numerical = tokenizer.texts_to_sequences(sequences)
sequences_padded = pad_sequences(sequences_numerical, maxlen=50, padding='pre')

vocab_size = len(tokenizer.word_index) + 1
X, y = sequences_padded[:, :-1], sequences_padded[:, -1]
y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=X.shape[1]))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(150, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

model.build(input_shape=(None, X.shape[1]))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X, y, batch_size=120, epochs=800)
model.save('Kazanjian_Moby_Model.keras')
print("Model saved successfully.")

with open('Kazanjian_Moby_Tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved successfully.")

print("Model training complete.")
