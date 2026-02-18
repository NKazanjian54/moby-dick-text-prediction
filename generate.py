import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from pickle import load

# Restore the tokenizer from the pickle file
with open('Kazanjian_Moby_Tokenizer.pickle', 'rb') as file:
    tokenizer = load(file)

# Load the model
model = load_model("Kazanjian_Moby_Model.keras")

# Use the model to generate the text
def generate_text(model, tokenizer, seq_len, seed_text, num_words):
    output_text = []

    input_text = seed_text  # Initial Seed Sequence

    for i in range(num_words):
        # Encode the input text sequence to numbers
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        # Ensure the input is of the correct length
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')

        # Predict the probability distribution for the next word
        predict_x = model.predict(pad_encoded)
        pred_word_ind = np.argmax(predict_x, axis=1)[0]

        # Retrieve the actual word from tokenizer
        pred_word = tokenizer.index_word[pred_word_ind]

        # Append to input text which is fed into the next iteration
        input_text += ' ' + pred_word
        output_text.append(pred_word)

    return ' '.join(output_text)

# Example seed text
input_sequence = "Call me Ishmael Some years ago"
words_to_generate = 200

# Generate text
result = generate_text(model, tokenizer, 25, input_sequence, words_to_generate)
print(result)
