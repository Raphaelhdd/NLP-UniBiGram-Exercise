import math
import bigram
import unigram

class LinearInterpolation:
    model_ = None
    unigram_model_ = None
    bigram_model_ = None
    lambda_unigram_ = None
    lambda_bigram_ = None

    def __init__(self, lambda_unigram, lambda_bigram, nlp, text=None, unigram_model=None, bigram_model=None):
        """
        Initializer for the LinearInterpolation class.

        Parameters:
        - lambda_unigram: Weight for the unigram model.
        - lambda_bigram: Weight for the bigram model.
        - nlp: A SpaCy language model.
        - text: Optional. List of texts for training the unigram and bigram models.
        - unigram_model: Pre-trained unigram model.
        - bigram_model: Pre-trained bigram model.
        """
        self.nlp_ = nlp
        self.lambda_unigram_ = lambda_unigram
        self.lambda_bigram_ = lambda_bigram
        if text:
            # If text is provided, train the unigram and bigram models
            self.unigram_model_ = unigram.Unigram(nlp, text).model_
            self.bigram_model_ = bigram.Bigram(nlp, text).model_
        elif unigram_model and bigram_model and not text:
            # If pre-trained models are provided, use them
            self.unigram_model_ = unigram_model
            self.bigram_model_ = bigram_model

    def predict(self, sentence):
        """
        Predict the log probability of a sentence using linear interpolation of unigram and bigram models.

        Parameters:
        - sentence: Input sentence.

        Returns:
        - prob: Log probability of the sentence.
        """
        prob = 0
        doc = self.nlp_(sentence)
        previous = "START"
        for token in doc:
            if token.is_alpha:
                lemma = token.lemma_
                if lemma in self.unigram_model_:
                    if lemma in self.bigram_model_[previous]:
                        prob += math.log(self.lambda_unigram_ * math.exp(self.unigram_model_[lemma]) +
                                         self.lambda_bigram_ * math.exp(self.bigram_model_[previous][lemma]))
                    else:
                        prob += math.log(self.lambda_unigram_ * math.exp(self.unigram_model_[lemma]))
                else:
                    return float('-inf')
                previous = lemma
        return prob

    def compute_perplexity(self, test_set):
        """
        Compute perplexity of a test set using the linear interpolation model.

        Parameters:
        - test_set: List of sentences in the test set.

        Returns:
        - perplexity: Perplexity of the test set.
        """
        total_words = 0
        total_prob = 0
        for sentence in test_set:
            total_prob += self.predict(sentence)
            total_words += len(sentence.split())
        l = 1 / total_words * total_prob
        return math.exp(-l)
