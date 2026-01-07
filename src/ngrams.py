""" Library that implements ngrams for Lab 4 of KIK-LG102 """

import nltk, math
from numpy import random

class Ngrams:

    def __init__(self, corpus_text):
        """ Constructor precomputes unigram, bigram, trigram and four frequencies from the text """
        self.unigram_freqs = nltk.FreqDist(corpus_text)
        self.bigram_freqs = nltk.ConditionalFreqDist(nltk.bigrams(corpus_text))
        self.trigram_freqs = \
            nltk.ConditionalFreqDist(((w1, w2), w3) for w1, w2, w3 in nltk.trigrams(corpus_text))
        self.fourgram_freqs = \
            nltk.ConditionalFreqDist(((w1, w2, w3), w4) for w1, w2, w3, w4 in nltk.ngrams(corpus_text, 4))
        
    def set_weights(self, zerogram_weight, unigram_weight, bigram_weight, trigram_weight, fourgram_weight):
        """ Set weights for the ngram mixture model (interpolated model) """
        self.weights = (zerogram_weight, unigram_weight, bigram_weight, trigram_weight, fourgram_weight)
        if sum(self.weights) != 1.0:
            print("Error: The weights summed together should be 1.0. The current value is", sum(self.weights))
            
    def p_zerogram(self):
        """ A zerogram model gives the same probability to every word. What the word actually
        is or what the history is is not relevant. """
        # Since not all words occur in the corpus data, we make the assumption that there are actually
        # the double number of words in reality compared to the words we have observed, so we multiply
        # the size of the vocabulary by 2.
        return 1.0 / (2 * self.unigram_freqs.B())

    def p_unigram(self, word):
        """ Unigram probability: only the word itself matters, not the history """
        if (word in self.unigram_freqs):
            # The unigram probability is the frequency of the word divided by the total number
            # of words
            return self.unigram_freqs[word] / self.unigram_freqs.N()
        # Else it is an out-of-vocabulary word, so the unigram probability is zero
        return 0.0

    def p_bigram(self, word, history):
        """ Bigram probability: the word matters and the last word of the history """
        if len(history) == 0 or history[-1] not in self.bigram_freqs:
            # The given history does not occur in the data, so the bigram probability is undefined
            return None
        if word in self.bigram_freqs[history[-1]]:
            # The bigram occurs in the data, so we can compute its probability
            return self.bigram_freqs[history[-1]][word] / self.bigram_freqs[history[-1]].N()
        # Else the bigram is not in the data, so the probability is zero
        return 0.0

    def p_trigram(self, word, history):
        """ Trigram probability: the word matters and the two preceding words """
        if len(history) < 2 or tuple(history[-2:]) not in self.trigram_freqs:
            # The given history does not occur in the data, so the trigram probability is undefined
            return None
        if word in self.trigram_freqs[tuple(history[-2:])]:
            # The trigram occurs in the data, so we can compute its probability
            return self.trigram_freqs[tuple(history[-2:])][word] / self.trigram_freqs[tuple(history[-2:])].N()
        # Else the trigram is not in the data, so the probability is zero
        return 0.0

    def p_fourgram(self, word, history):
        """ Fourgram probability: the word matters and the three preceding words """
        if len(history) < 3 or tuple(history[-3:]) not in self.fourgram_freqs:
            # The given history does not occur in the data, so the fourgram probability is undefined
            return None
        if word in self.fourgram_freqs[tuple(history[-3:])]:
            # The fourgram occurs in the data, so we can compute its probability
            return self.fourgram_freqs[tuple(history[-3:])][word] / self.fourgram_freqs[tuple(history[-3:])].N()
        # Else the fourgram is not in the data, so the probability is zero
        return 0.0

    def p_ngram(self, word, history):
        """ Compute probability of a word given a history of words """
        
        # Return weighted mixture of zerogram, unigram, bigram, trigram and fourgram probabilities
        # (If the bigram, trigram or fourgram probs are undefined because the requested history does not
        # exist in the data, then normalize such that only the defined probabilities count.)
        p = self.weights[0] * self.p_zerogram() + self.weights[1] * self.p_unigram(word)
        weight_sum = self.weights[0] + self.weights[1]
        
        p_bigram = self.p_bigram(word, history)
        if p_bigram is not None:
            p += self.weights[2] * self.p_bigram(word, history)
            weight_sum += self.weights[2]
            
        p_trigram = self.p_trigram(word, history)
        if p_trigram is not None:
            p += self.weights[3] * self.p_trigram(word, history)
            weight_sum += self.weights[3]

        p_fourgram = self.p_fourgram(word, history)
        if p_fourgram is not None:
            p += self.weights[4] * self.p_fourgram(word, history)
            weight_sum += self.weights[4]
            
        if weight_sum == 0.0:
            return 0.0

        return p / weight_sum

    def perplexity(self, test_text):
        """ Compute perplexity of the model on a test text (= list of words) """
        logprob = 0.0
        history = []
        for word in test_text:
            p = self.p_ngram(word, history)
            if p == 0.0:
                return math.inf
            logprob -= math.log(p)
            history.append(word)
        return math.exp(logprob / len(test_text))
    
    def pdist_ngram(self, history):
        """ Compute the full probability distribution for the next word following the history """
        return { w: self.p_ngram(w, history) for w in self.unigram_freqs.keys() }
    
    def sorted_pdist_ngram(self, history):
        """ Produce a probability distribution as a list of tuples (word, prob) with highest probs first
        and descending """
        pdist = self.pdist_ngram(history)
        return sorted([(w, p) for w, p in pdist.items()], key=lambda wp: wp[1], reverse=True)
    
    def pdist_ngram_normalized(self, history):
        """ Make a probability distribution that does not leak (because of OOVs), because random.choice
        is so picky about it """
        pdist = self.pdist_ngram(history)
        ptot = sum(pdist.values())
        return { w: pdist[w] / ptot for w in pdist.keys() }
    
    def generate_sentence(self, start=[], max_len=100):
        """ Generate a random sentence that optionally starts with a given list of words. The sentence
        ends when the first period is encountered or when the length of the sentence is 100 words. """
        # Find out a history for your next sentence: it is some ngram that occurs in the corpus
        # that either matches the end of the given "start" list or that ends in a period
        preceding3grams = []  # initialize as no match at all
        if len(start) >= 3:
            # Find all fourgrams that start with the desired trigram from the "start" list
            preceding3grams = \
                [w1 + " " + w2 + " " + w3 for w1, w2, w3 in self.fourgram_freqs.keys() \
                     if w1 == start[-3] and w2 == start[-2] and w3 == start[-1]]
        if len(start) >= 2 and len(preceding3grams) == 0:
            # Find all fourgrams that contain the desired bigram at the end of the "start" list
            preceding3grams = \
                [w1 + " " + w2 + " " + w3 for w1, w2, w3 in self.fourgram_freqs.keys() \
                     if w2 == start[-2] and w3 == start[-1]]
        if len(start) >= 1 and len(preceding3grams) == 0:
            # Find all fourgrams that contain the desired unigram at the end of the "start" list
            preceding3grams = \
                [w1 + " " + w2 + " " + w3 for w1, w2, w3 in self.fourgram_freqs.keys() if w3 == start[-1]]
        if len(preceding3grams) == 0:
            # The caller does not ask for any particular history, or we did not find any match for what
            # the caller desires, so try find all fourgrams that have a period in penultimate position
            preceding3grams = \
                [w1 + " " + w2 + " " + w3 for w1, w2, w3 in self.fourgram_freqs.keys() if w3 == '.']
        if len(preceding3grams) == 0:
            # Still no luck: just pick the trigrams that start any of the fourgrams in the corpus
            preceding3grams = [w1 + " " + w2 + " " + w3 for w1, w2, w3 in self.fourgram_freqs.keys()]
            
        history = random.choice(preceding3grams, 1)[0].split(" ")
        for i in range(max_len):
            pdist = self.pdist_ngram_normalized(history)
            word = random.choice(list(pdist.keys()), 1, p=list(pdist.values()))[0]
            history.append(word)
            if word == ".":
                break
        return [*start, *history[3:]]
