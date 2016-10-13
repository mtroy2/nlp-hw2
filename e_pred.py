import random
import tkinter as tk
import nltk as nltk
import time
import math
class Uniform(object):
    """Barebones example of a language model class."""

    def __init__(self,ngrams):
        self.grams =     {}
        self.ngrams =    ngrams
        self.prob_dict = {}
        self.state = ""
        self.chars = 'qwertyuiopasdfghjklzxcvbnm,. '
        self.time = 0
        for i in range(0,ngrams):  
            self.grams[i+1]     = {}
            self.prob_dict[i+1] = {}
    def train(self, filename):
        """Train the model on a text file."""
        train_file = open(filename)
        t_file = train_file.read()
 
        t_file = t_file.replace('\n',' ')
            
        # form all ngrams from line
        for i in range(1,self.ngrams+1):
            ngrams_list = nltk.ngrams('恩'*(i-1) + t_file, i)
            # add to ngram dictionary
            for entry in ngrams_list:
                if len(entry) == 1:
                    context = ''
                else:
                    context = ''.join(entry[0:len(entry)-1])
                letter = entry[-1]
                
                if context in self.grams[i].keys():
                    if letter in self.grams[i][context].keys():
                        self.grams[i][context][letter] += 1.
                    else:
                        self.grams[i][context][letter] = 1.
                else:
                    self.grams[i][context] = {}
                    self.grams[i][context][letter] = 1
                    # add to dictionary / build vocabulary

        return
    def compute_dev(self, filename):
        """ Read in a file, predict most probably char for each char position in file """

        dev_set = open(filename)
        
        ten_pos = dev_set.read(10)
        ten_pos = ten_pos.replace('\n',' ')
        # append start symbols handled in self.start()
        self.compute(ten_pos, 'dev set ', 't')
        return
    def compute_test(self,filename):

        test_set = open(filename)

        f_str = test_set.read()
        f_str = f_str.replace('\n',' ')
        self.test_file = f_str
        self.compute(f_str, 'test set ')
        return
    def compute(self,f_string,name, prin = 'f'):
        correct = 0
        incorrect = 0
        for i,char in enumerate(f_string):
            #if char == '\n':
                #self.start()    
            guess,prob = self.predict()
            if prin == 't':
                print('Guess<' + guess + '>, probability<' + str(prob) + '>')
            if guess == char:
                correct += 1
            else:
                incorrect += 1
            self.read(char)
        print( 'Accuracy on ' + name + '= ' + str( correct / (incorrect + correct) ) )
        return
    def compute_perplexity(self,filename):
        i_file = open(filename)
        in_str = i_file.read()
        in_str = in_str.replace('\n',' ')
        in_str = '恩'*(self.ngrams-1) + in_str
        in_grams = nltk.ngrams(in_str,self.ngrams)
        perp = 0
        summation = 0
        for gram in in_grams:
            char = gram[-1]
            context = ''.join(gram[:-1])
            summation += math.log(self.recurse_smooth(char, context,self.ngrams))
        summation = summation* (-1/len(in_str))
        perp = math.exp(summation)
        print("Perplexity on set is " + str(perp))
    def predict(self):
        """ Returns char with highest probability given current state """
        #print("state<" + self.state + ">")

        max_prob = 0.
        ret_pair = ['',0.]
     
        if self.ngrams == 1:
            context = "" 
        else:
            context = self.state[-(self.ngrams-1):]
        for char in self.chars:

            prob = self.recurse_smooth(char, context, self.ngrams)

            if prob > max_prob:
                max_prob = prob
                ret_pair = [char,prob]
        return ret_pair[0],ret_pair[1]

    def recurse_smooth(self,char, context,dict_num):
        """ Computes recursively p(w | u) """

        if context in self.prob_dict[dict_num].keys():
            if char in self.prob_dict[dict_num][context].keys():
                return self.prob_dict[dict_num][context][char]
        else:
            self.prob_dict[dict_num][context] = {}
        
        # unigram
        if context == "":
            self.prob_dict[dict_num][context][char] = self.unigram_prob(char) 
            return self.prob_dict[dict_num][context][char]

      
        c_u_dot,lam = self.lambda_compute(context, dict_num)
        if context in self.grams[dict_num].keys():
            if char in self.grams[dict_num][context].keys():
                c_u_w = self.grams[dict_num][context][char]
            else:
                c_u_w = 0.
        else:
            c_u_w = 0
        if len(context) == 1:
            new_context = ""
        else:
            new_context = context[1:]
        if c_u_dot == 0:
            self.prob_dict[dict_num][context][char] =(0 + (1.-lam)*self.recurse_smooth(char,new_context,dict_num-1))   
            return self.prob_dict[dict_num][context][char] 
        else: 
            self.prob_dict[dict_num][context][char] =((lam*c_u_w/c_u_dot) + (1.-lam)*self.recurse_smooth(char,new_context,dict_num-1))
            return self.prob_dict[dict_num][context][char]

    def unigram_prob(self,char):
        """ Returns unigram prob of one single char in model """
        char_count = sum(self.grams[1][""].values())
        vocab_len = len(self.grams[1][""].keys())
        lambda_char = char_count / (char_count + vocab_len)
        count = self.grams[1][""][char]

        return ((lambda_char * count / char_count) + (1-lambda_char)*(1./vocab_len))

    def lambda_compute(self, context, dict_num):
        """ Computes lambda value necessary for witten bell smoothing """
        # compute c(u o )
        if context in self.grams[dict_num].keys():
            c_u_dot = sum(self.grams[dict_num][context].values())
            n_plus_u = len(self.grams[dict_num][context].keys())
        else:
            c_u_dot  = 0
            n_plus_u = 0

        if (c_u_dot + n_plus_u) == 0:
            lam = 0
        else:
            lam = c_u_dot / (c_u_dot + n_plus_u+1)
        
        return c_u_dot , lam
    

    # The following two methods make the model work like a finite
    # automaton.
    def start(self):
        """Resets the state."""
        self.state = "恩"*(self.ngrams-1)
        return

    def read(self, w):
        """Reads in character w, updating the state."""
        self.state = self.state + w
        return

    # The following two methods add probabilities to the finite automaton.





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='train')
    parser.add_argument(dest='dev')
    parser.add_argument(dest='test')
    parser.add_argument(dest='ngrams')
    args = parser.parse_args()

    ##### Replace this line with an instantiation of your model #####
    try:
    
        ngrams = int(args.ngrams)
        m = Uniform(ngrams)
        m.train(args.train)
        m.start()
        m.compute_dev(args.dev)
        m.compute_perplexity(args.test)
        m.compute_test(args.test)   
    except ValueError:
        print("Error: Invalid ngram argument. Please enter an integer argument")
        
