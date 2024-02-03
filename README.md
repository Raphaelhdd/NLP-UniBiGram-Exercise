# NLP-UniBiGram-Exercise

This repository contains an implementation of simple unigram and bigram language models, along with linear interpolation
between them, for Natural Language Processing (NLP). The models are trained on text data from the Wikitext dataset using
the SpaCy library for language processing and Huggingface's datasets package for acquiring the data.

## Requirements

To run the code in this repository, you need to install the following packages:

- SpaCy (for language processing)
- Huggingface's datasets (for acquiring data)

You can install these dependencies using:

pip install spacy
pip install datasets

**Usage:**

1. Clone the Repository:
git clone https://github.com/your-username/NLP-UniBiGram-Exercise.git
cd NLP-UniBiGram-Exercise

2. Run the code:
python main.py


## Project Structure

The repository is organized as follows:

- **`unigram.py`**: Contains the implementation of the Unigram class for the unigram language model.
- **`bigram.py`**: Contains the implementation of the Bigram class for the bigram language model.
- **`linear_interpolation.py`**: Contains the implementation of the LinearInterpolation class for linear interpolation between unigram and bigram models.
- **`main.py`**: The main script that utilizes the models and answers specific questions about the models.


## Explanation of Classes:

### Unigram Class:
- Represents a simple unigram language model.
- Implements training based on maximum likelihood estimators.
- Provides methods for:
  - Predicting probabilities.
  - Computing perplexity.

### Bigram Class:
- Represents a simple bigram language model.
- Implements training based on maximum likelihood estimators with linear interpolation.
- Provides methods for:
  - Predicting probabilities.
  - Predicting the next word.
  - Computing perplexity.

### LinearInterpolation Class:
- Represents a language model that performs linear interpolation between unigram and bigram models.
- Allows customization of interpolation weights.
- Provides methods for:
  - Predicting probabilities.
  - Computing perplexity.


## Questions Answered in `main.py`:

### 1. Prediction with Bigram Model:

This part of the code addresses the task of predicting the most probable word following a given sentence using the bigram language model. The `Bigram` class is utilized to achieve this. It calculates and outputs the word that is most likely to follow the input sentence based on the bigram model's probabilities.

### 2. Bigram Model Probabilities:

In this section, the script calculates the probability of two specified sentences using the bigram language model. The `Bigram` class is employed to compute the likelihood of each sentence occurring according to the bigram model. The results are then printed, providing insights into the model's estimation of the likelihood of these sentences.

### 3. Compute the Perplexity of a Test Set:

Perplexity is a measure commonly used to evaluate language models. Here, the script calculates the perplexity of a test set containing two sentences using the bigram model. The perplexity score helps assess how well the language model predicts the test set. A lower perplexity indicates better model performance.

### 4. Linear Interpolation Model:

This part of the code focuses on training a new language model using linear interpolation smoothing between the bigram and unigram models. The `LinearInterpolation` class is employed to create this hybrid model, allowing for a combination of both unigram and bigram probabilities. The script then computes the probability and perplexity of the same two sentences using this newly created linear interpolation model.

### License:

The project is licensed under the MIT License. The LICENSE file contains details about the terms and conditions under which the code can be used, modified, and distributed. It is important to review and comply with the license when working with or utilizing the code from this repository.

