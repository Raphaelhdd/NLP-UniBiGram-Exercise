import math

class Bigram:
    model_ = None
    nlp_ = None

    def __init__(self, nlp, text=None):
        """
        Initializer for the Bigram class.

        Parameters:
        - nlp: A SpaCy language model.
        - text: Optional. List of texts for training the bigram model.
        """
        self.nlp_ = nlp
        if text:
            model = dict()
            word_appearance = dict()
            for line in text:
                previous = "START"
                doc = nlp(line['text'])
                for token in doc:
                    if token.is_alpha:
                        lemma = token.lemma_
                        if previous not in model:
                            model[previous] = dict()
                            word_appearance[previous] = 0
                        word_appearance[previous] += 1
                        if lemma not in model[previous]:
                            model[previous][lemma] = 0
                        model[previous][lemma] += 1
                        previous = lemma
            self.model_ = {
                lemma0: {lemma1: math.log(appearance1 / word_appearance[lemma0]) for lemma1, appearance1 in
                         dic.items()} for lemma0, dic in model.items()}

    def predict(self, sentence):
        """
        Predict the log probability of a sentence using the bigram model.

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
                if previous in self.model_ and lemma in self.model_[previous]:
                    prob += self.model_[previous][lemma]
                else:
                    return float('-inf')
                previous = lemma
        return prob

    def next_word(self, sentence):
        """
        Predict the next word in a sentence using the bigram model.

        Parameters:
        - sentence: Input sentence.

        Returns:
        - predicted_word: The predicted next word.
        """
        doc = self.nlp_(sentence)
        lemma = "START"
        for i in range(len(doc) - 1, -1, -1):
            if i == 0:
                lemma = "START"
            if doc[i].is_alpha:
                lemma = doc[i].lemma_
                break
            i -= 1
        max_prob = float('-inf')
        predicted_word = None
        if lemma in self.model_:
            for next_lemma, prob in self.model_[lemma].items():
                if prob > max_prob:
                    max_prob = prob
                    predicted_word = next_lemma
        return predicted_word

    def compute_perplexity(self, test_set):
        """
        Compute perplexity of a test set using the bigram model.

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
