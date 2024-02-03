import spacy
from datasets import load_dataset
import bigram
import linear_interpolation
import unigram

if __name__ == '__main__':
    # Load SpaCy language model
    nlp = spacy.load('en_core_web_sm')

    # Load Wikitext dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")

    # Train unigram and bigram models
    unigram_model = unigram.Unigram(nlp, dataset)
    bigram_model = bigram.Bigram(nlp, dataset)

    # Question 2
    sentence = "I have a house in"
    word = bigram_model.next_word(sentence)
    print("Question 2:")
    print(f"Using the bigram model, the most probable word predicted by the model after '{sentence}' is: {word}\n")

    # Question 3:
    first_sentence = "Brad Pitt was born in Oklahoma"
    second_sentence = "The actor was born in USA"
    probability_first_sentence = bigram_model.predict(first_sentence)
    probability_second_sentence = bigram_model.predict(second_sentence)
    print("Question 3.a:")
    print(f"The probability using the bigram model of the sentence '{first_sentence}' is: {probability_first_sentence}")
    print(f"The probability using the bigram model of the sentence '{second_sentence}' is: {probability_second_sentence}\n")

    test_set = [first_sentence, second_sentence]
    perplexity_bigram_set = bigram_model.compute_perplexity(test_set)
    print("Question 3.b:")
    print(f"Perplexity of the test set using bigram model is: {perplexity_bigram_set}\n")

    # Question 4:
    model_linear_interpolation = linear_interpolation.LinearInterpolation(lambda_unigram=1 / 3, lambda_bigram=2 / 3,
                                                                          nlp=nlp, unigram_model=unigram_model.model_,
                                                                          bigram_model=bigram_model.model_)
    probability_linear_first_sentence = model_linear_interpolation.predict(first_sentence)
    probability_linear_second_sentence = model_linear_interpolation.predict(second_sentence)
    perplexity_linear_set = model_linear_interpolation.compute_perplexity(test_set)
    print("Question 4:")
    print(f"The probability using the linear interpolation model of the sentence '{first_sentence}' is: "
          f"{probability_linear_first_sentence}")
    print(f"The probability using the linear interpolation model of the sentence '{second_sentence}' is: "
          f"{probability_linear_second_sentence}\n")
    print(f"Perplexity of the test set using linear interpolation model is: {perplexity_linear_set}\n")
