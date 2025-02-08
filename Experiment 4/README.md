# Experiment-4: Poetry Generation using RNN

## Objective
To generate poetry using a Recurrent Neural Network (RNN) trained on a dataset of poems. The model is implemented using TensorFlow and Keras, utilizing LSTM layers for sequence prediction.

## Dataset
- **Source:** A dataset of 100 poems stored in `poems-100.csv`.
- **Preprocessing:**
  - Text normalization: converting to lowercase, removing newlines.
  - Tokenization using `RegexpTokenizer`.
  - Stopword removal (if necessary).
  - Encoding words into one-hot vectors.

## Model Architecture
- **Input Layer:** Sequence of one-hot encoded words.
- **LSTM Layers:**
  - First LSTM layer with 256 units and `return_sequences=True`.
  - Second LSTM layer with 128 units.
- **Dense Layer:** Output layer with softmax activation for word prediction.
- **Optimizer:** Adam.
- **Loss Function:** Categorical Crossentropy.
- **Evaluation Metric:** Accuracy.

## Training
- **Epochs:** 100
- **Batch Size:** 64
- **Performance Metrics:**
  - Final training accuracy: **96.50%**
  - Final training loss: **0.1620**

## Results
A plot of training loss and accuracy over epochs was generated for visualization.

## Text Generation
- **Prediction Function:** Predicts the next word based on a given sequence.
- **Text Generator:** Generates poetry by iteratively predicting and appending words.
- **Creativity Parameter:** Controls randomness in word selection.

## Output Example
Input: _"how do I love thee let me count the ways"_

Generated Text (100 words, creativity=5):
```
how do i love thee let me count the ways of time and dreams through golden light
as whispers dance where stars align the morning sings in melodies bright and clear...
```

## Model Saving
- Trained weights are saved as `rnn_poem.weights.h5`.

## Conclusion
The model successfully generates poetry-like sequences. Future improvements may include:
- Using a larger dataset.
- Experimenting with different architectures (e.g., Transformer-based models).
- Fine-tuning hyperparameters for better coherence.