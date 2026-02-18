# Moby Dick LSTM Text Generator

A deep learning project that trains a character-level language model on Herman Melville's *Moby Dick* and generates new text in a similar style. Built using Python, TensorFlow/Keras, and spaCy.

---

## How It Works

This project uses a **Long Short-Term Memory (LSTM)** neural network — a type of recurrent neural network (RNN) well-suited for learning sequential patterns in text.

### Training Pipeline (`main.py`)

1. **Text Cleaning** — The raw text is lowercased, stripped of punctuation and extra whitespace, and processed through spaCy to produce a clean list of tokens (words).

2. **Sequence Creation** — The token list is broken into overlapping sequences of 50 words each. Each sequence becomes a training sample where the model learns to predict the next word given the previous 50.

3. **Tokenization** — Keras's `Tokenizer` converts each word into a unique integer, building a vocabulary from the entire text.

4. **Model Architecture** — The neural network consists of:
   - An **Embedding layer** that maps each word integer to a dense 100-dimensional vector, capturing semantic relationships between words
   - Two **LSTM layers** (150 units each) that learn patterns across word sequences over time
   - A **Dense layer** with ReLU activation for additional feature learning
   - A final **Dense output layer** with Softmax activation that outputs a probability distribution over the entire vocabulary — the word with the highest probability becomes the predicted next word

5. **Training** — The model is compiled with categorical cross-entropy loss and the Adam optimizer, then trained over multiple epochs. The trained model and tokenizer are saved to disk for later use.

### Text Generation (`generate.py`)

The saved model and tokenizer are loaded, then given a **seed phrase** (starting text). The model predicts the most likely next word, appends it to the input, and repeats — generating new text word by word.

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### Install Dependencies

```bash
pip install tensorflow spacy numpy
python -m spacy download en_core_web_sm
```

---

## Usage

### Step 1 — Train the Model

Run `main.py` to process the text, train the LSTM, and save the model and tokenizer:

```bash
python main.py
```

This will generate two files:
- `Kazanjian_Moby_Model.keras` — the trained model
- `Kazanjian_Moby_Tokenizer.pickle` — the fitted tokenizer

> Note: Training is computationally intensive. On CPU this may take a significant amount of time depending on your hardware and epoch count.

### Step 2 — Generate Text

Once training is complete, run `generate.py` to produce new text:

```bash
python generate.py
```

The default seed phrase is `"Call me Ishmael Some years ago"` and generates 200 words. Both values can be changed at the bottom of `generate.py`:

```python
input_sequence = "Call me Ishmael Some years ago"
words_to_generate = 200
```

---

## Project Structure

```
├── main.py                      # Training script
├── generate.py                  # Text generation script
├── moby_dick_full.txt           # Full training text
├── moby_dick_four_chapters.txt  # Smaller training subset
├── moby_dick_short.txt          # Minimal text sample
└── README.md
```

---

## Tech Stack

- **Python** — Core language
- **TensorFlow / Keras** — Neural network architecture and training
- **spaCy** — NLP preprocessing and tokenization
- **NumPy** — Numerical operations

---

## Author

Nick Kazanjian — [GitHub](https://github.com/NKazanjian54)
