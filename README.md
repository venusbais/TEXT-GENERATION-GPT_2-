
# GPT-2 Text Generation Project

## Overview

Welcome to the GPT-2 Text Generation project! This project leverages the powerful GPT-2 model from OpenAI to generate coherent and contextually relevant text based on user inputs. The project includes features like interactive text generation, sentiment analysis, and visualization of tokenized sentences.

## Features

- **Interactive Text Generation**: Input your own sentences and watch GPT-2 generate text.
- **Sentiment Analysis**: Get immediate insights into the sentiment of the generated text using NLTK’s VADER.
- **Visualization**: Visualize how the model tokenizes and processes your input.

## Why This Project Matters

Artificial Intelligence is transforming how we interact with text and language. This project demonstrates the potential of AI to generate meaningful and insightful text, showcasing advanced techniques in natural language processing.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Jupyter Notebook or Google Colab

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/GPT-2-Text-Generation-Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd GPT-2-Text-Generation-Project
   ```
3. Install the required libraries:
   ```bash
   pip install transformers torch matplotlib nltk
   ```
4. Download NLTK data:
   ```python
   import nltk
   nltk.download('vader_lexicon')
   ```

### Running the Project

Open the Jupyter Notebook or Google Colab and run the provided code cells to interact with the model.

## Key Components

### Tokenization

Utilizes Byte Pair Encoding (BPE) for efficient text processing.

### Self-Attention Mechanism

Uses self-attention to focus on different parts of the input sequence, ensuring contextual understanding.

### Beam Search

Enhances the quality of generated text by exploring multiple sequences.

## Example Usage

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Initialize model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)

# Input sentence
sentence = input("Enter a sentence to generate text: ")
numeric_ids = tokenizer.encode(sentence, return_tensors='pt')

# Generate text
result = model.generate(numeric_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
generated_text = tokenizer.decode(result[0], skip_special_tokens=True)
print(generated_text)

# Sentiment analysis
sid = SentimentIntensityAnalyzer()
scores = sid.polarity_scores(generated_text)
print("Sentiment Scores:", scores)

# Visualization
tokens = tokenizer.convert_ids_to_tokens(numeric_ids[0])
plt.figure(figsize=(15, 5))
plt.bar(range(len(tokens)), [1] * len(tokens), tick_label=tokens)
plt.show()
```

## Interesting References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://openai.com/research/language-models-are-unsupervised-multitask-learners)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

## Contributing

Feel free to fork this repository and contribute by submitting pull requests. Any enhancements or bug fixes are welcome!


## Connect with Me

Let’s connect and explore the endless possibilities with AI!

- LinkedIn: https://www.linkedin.com/in/venus-bais-datascience/
- GitHub: https://github.com/venusbais

---

