import math

class Unigram:
    def __init__(self, nlp, text=None):
        """
        Initializer for the Unigram class.

        Parameters:
        - nlp: A SpaCy language model.
        - text: Optional. List of texts for training the unigram model.
        """
        self.nlp_ = nlp
        self.model_ = None
        if text:
            self.build_model(text)

    def build_model(self, text):
        """
        Build the unigram model based on the provided text.

        Parameters:
        - text: List of texts for training the unigram model.
        """
        model = dict()
        num_words = 0
        for line in text:
            doc = self.nlp_(line['text'])
            for token in doc:
                if token.is_alpha:
                    lemma = token.lemma_
                    num_words += 1
                    if lemma not in model:
                        model[lemma] = 0
                    model[lemma] += 1

        # Calculate log probabilities and store in the model
        self.model_ = {lemma: math.log(appearance / num_words) for lemma, appearance in model.items()}

    def predict(self, sentence):
        """
        Predict the probability of a sentence using the unigram model.

        Parameters:
        - sentence: Input sentence.

        Returns:
        - prob: Log probability of the sentence.
        """
        prob = 0
        doc = self.nlp_(sentence)
        for token in doc:
            if token.is_alpha:
                lemma = token.lemma_
                # Use get() to handle the case where the lemma is not in the model
                prob += self.model_.get(lemma, 0.0)
        return prob
