import random
import tkinter as tk
from enum import Enum
import nltk as nltk


class Ngrams(Enum):
    unigram = 1
    bigram = 2
    trigram = 3
    fourgram = 4
    fivegram = 5
    sixgram = 6
    sevengram = 7
    eightgram = 8
class Uniform(object):
    """Barebones example of a language model class."""

    def __init__(self,ngrams):
        self.grams =     {}
        self.ngrams =    ngrams
        self.prob_dict = {}

        for i in range(0,ngrams):  
            self.grams[i+1]     = {}
            self.prob_dict[i+1] = {}

    def train(self, filename):
        """Train the model on a text file."""

        for line in open(filename):
            line.replace('\n',' ')
            line_tokens = []
            for w in line:
                # tokenize line
                line_tokens.append(w)
            # form all ngrams from line
            for i in range(1,self.ngrams+1):
                ngrams_list = nltk.ngrams(line_tokens, i)
                # add to ngram dictionary
                for entry in ngrams_list:
                    if entry not in self.grams[i]:
                        # add to dictionary / build vocabulary
                        self.grams[i][entry] = 1.
                    else:
                        self.grams[i][entry] += 1.
        self.probs()

    # The following two methods make the model work like a finite
    # automaton.
    def start(self):
        """Resets the state."""
        pass

    def read(self, w):
        """Reads in character w, updating the state."""
        pass

    # The following two methods add probabilities to the finite automaton.



    def prob(self, w, ngram_vocab):
        """Returns the probability of the next character being w given the
        current state."""
        return 1/(len(self.vocab)+1) # +1 for <unk>

    def probs(self):
        """Returns a dict mapping from all characters in the vocabulary to the
probabilities of each character."""
        for dict_number,gram_dict in self.grams.items():
            for g, count in gram_dict.items():
                self.prob_dict[dict_number][g] = count / (sum(gram_dict.values())+1)
       
        for key,value in self.prob_dict[1].items():
            print(str(key) + ' -> ' + str(value))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='train')
    args = parser.parse_args()

    ##### Replace this line with an instantiation of your model #####
    ngrams = 2
    m = Uniform(ngrams)
    m.train(args.train)
    m.start()

    #root = tk.Tk()
    #app = Application(m, master=root)
    #app.mainloop()
    #root.destroy()
