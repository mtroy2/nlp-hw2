import random
import tkinter as tk
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
            line = '\t' + line + '\a'
            # form all ngrams from line
            for i in range(1,self.ngrams+1):
                ngrams_list = nltk.ngrams(line, i)
                # add to ngram dictionary
                for entry in ngrams_list:
                    if entry not in self.grams[i]:
                        # add to dictionary / build vocabulary
                        self.grams[i][entry] = 1.
                    else:
                        self.grams[i][entry] += 1.
        self.probs()
        for val, prob in self.prob_dict[2].items():
            print ( 'prob of ' + str(val) + ' = ' + str(prob) )
        for i, dic in self.prob_dict.items():
            print("sum of probs for " + str(i) + " grams = " + str(sum(dic.values())))
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
        #compute probs of unknown ngrams
        #for i,ngram_dict in self.grams.items():
        #   ngram_dict['unk'] = list(ngram_dict.values()).count(1.) / sum(ngram_dict.values())
        self.smooth_one_gram()
        self.witten_bell_smooth()

    def smooth_one_gram(self):
        """Returns smoothed 1-gram model"""
        char_count = sum(self.grams[1].values())
        vocab_len = len(self.grams[1].keys())
        lambda_char = char_count / (char_count + vocab_len)
        for char, count in self.grams[1].items():
            self.prob_dict[1][char] = (lambda_char * count / char_count) + (1 - lambda_char)*(1./vocab_len) 
        return
    
    def witten_bell_smooth(self):

        # start with 2-grams and continue upwards
        # compute lambda
        for dict_num, count_dict in self.grams.items():
            
            if dict_num != 1.:
                
                for gram, count in count_dict.items():
                    # get lambda value - input is all chars besides last char of ngram
                    package = self.lambda_compute(gram[0:len(gram)-1], dict_num)
                    c_u_dot = package[0]
                    lam = package[1]
                    c_u_w = count
                    recurse_p_wu = self.prob_dict[dict_num - 1][gram[1:]]
                    self.prob_dict[dict_num][gram] = (lam*c_u_w/c_u_dot) + (1.-lam)*recurse_p_wu

    def lambda_compute(self, u, dict_num):
        """ Computes lambda value necessary for witten bell smoothing """
        # compute c(u o )
        scan_dict = self.grams[dict_num]
        c_u_dot = 0.
        n_plus_u = 0.
        for gram, count in scan_dict.items():
            if gram[0:len(u)] == u:
                c_u_dot += count ## is this actually supposed to be the count?
                n_plus_u += 1.
        lam = c_u_dot / (c_u_dot + n_plus_u)
        return c_u_dot, lam
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='train')
    parser.add_argument(dest='ngrams')
    args = parser.parse_args()

    ##### Replace this line with an instantiation of your model #####
    #try:
    
    ngrams = int(args.ngrams)
    m = Uniform(ngrams)
    m.train(args.train)
    m.start()

    #except ValueError:
    #print("Error: Invalid ngram argument. Please enter an integer argument")
        

    #root = tk.Tk()
    #app = Application(m, master=root)
    #app.mainloop()
    #root.destroy()
