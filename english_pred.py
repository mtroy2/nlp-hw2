import random
import tkinter as tk
import nltk as nltk
class Uniform(object):
    """Barebones example of a language model class."""

    def __init__(self,ngrams):
        self.grams =     {}
        self.ngrams =    ngrams
        self.prob_dict = {}
        self.state = ""
        self.chars = 'qwertyuiopasdfghjklzxcvbnm,. '

        for i in range(0,ngrams):  
            self.grams[i+1]     = {}

    def train(self, filename):
        """Train the model on a text file."""

        for line in open(filename):

            # form all ngrams from line
            for i in range(1,self.ngrams+1):
                line.replace('\n',' ')
                line = '`'*(i-1) + line
                ngrams_list = nltk.ngrams(line, i)
                # add to ngram dictionary
                for entry in ngrams_list:
                    if entry not in self.grams[i]:
                        # add to dictionary / build vocabulary
                        self.grams[i][''.join(entry)] = 1.
                    else:
                        self.grams[i][entry] += 1.
        
    def compute_dev(self, filename):
        """ Read in a file, predict most probably char for each char position in file """

        dev_set = open(filename)
        
        ten_pos = dev_set.read(10)
        ten_pos.replace('\n',' ')
        # append start symbols handled in self.start()
        self.compute(ten_pos)
    def compute_test(self,filename):

        test_set = open(filename)

        f_str = test_set.read()
        f_str.replace('\n',' ')
        self.compute(f_str)
    def compute(self,f_string):
        correct = 0
        incorrect = 0
        for i,char in enumerate(f_string):
            guess,prob = self.predict()
            print('pos< ' + str(i) + ' >, guess< ' + guess + ' >, prob< ' + str(prob) + ' >')
            if guess == char:
                correct += 1
            else:
                incorrect += 1
            print("Current accuracy< " + str(correct/(correct+incorrect) ) + " >")
            self.read(char)
        print( 'Accuracy = ' + str( correct / (incorrect + correct) ) )
            
    def predict(self):
        """ Returns char with highest probability given current state """
        #print("state<" + self.state + ">")

        max_prob = 0.
        ret_pair = ['',0.]
        for char in self.chars:
            if self.ngrams == 1:
                prior = "" 
            else:
                prior = self.state[(len(self.state))-(self.ngrams-1):]
           
            prob = self.recurse_smooth(char, prior, self.ngrams)
            
            #print("Char<" + char + ">, Prob<" + str(prob) + ">")
            if prob > max_prob:
                max_prob = prob
                ret_pair = [char,prob]
        return ret_pair[0],ret_pair[1]
    def recurse_smooth(self,char, prior,dict_num):
        """ Computes recursively p(w | u) """

        if (prior+char) in self.prob_dict.keys():
            return self.prob_dict[prior+char]

        if len(prior) == 0:
            self.prob_dict[char] = self.unigram_prob(char) 
            return self.prob_dict[char]

        c_u_dot,lam = self.lambda_compute(prior, dict_num)
        if (prior+char) not in self.grams[dict_num].keys():
            c_u_w = 0
        else:
            c_u_w = self.grams[dict_num][prior + char]

        if len(prior) == 1:
            new_prior = ""
        else:
            new_prior = prior[1:]
        if c_u_dot == 0:
                self.prob_dict[prior+char] =(0 + (1.-lam)*self.recurse_smooth(char,new_prior,dict_num-1))   
                return self.prob_dict[prior+char] 
        else: 
                self.prob_dict[prior+char] =((lam*c_u_w/c_u_dot) + (1.-lam)*self.recurse_smooth(char,new_prior,dict_num-1))
                return self.prob_dict[prior+char]

    def unigram_prob(self,char):
        """ Returns unigram prob of one single char in model """

        char_count = sum(self.grams[1].values())
        vocab_len = len(self.grams[1].keys())
        lambda_char = char_count / (char_count + vocab_len)
        count = self.grams[1][char]

        return ((lambda_char * count / char_count) + (1-lambda_char)*(1./vocab_len))

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
        if (c_u_dot + n_plus_u) == 0:
            lam = 0
        else:
            lam = c_u_dot / (c_u_dot + n_plus_u)
        return c_u_dot, lam
        


    # The following two methods make the model work like a finite
    # automaton.
    def start(self):
        """Resets the state."""
        self.state = '`'*(self.ngrams-1)
        return

    def read(self, w):
        """Reads in character w, updating the state."""
        self.state += w
        return

    # The following two methods add probabilities to the finite automaton.



    def prob(self, w):
        """Returns the probability of the next character being w given the
        current state."""
        considered_text = self.state[ (len(self.state) - self.ngrams -1): ]
        c_u_dot, lam = self.lambda_compute(considered_text,self.ngrams)

        if (considered_text + w) in self.grams[self.ngrams].keys():
            c_u_w  = self.grams[self.ngrams][considered_text + w]
        else:
            c_u_w = 0
        if c_u_dot != 0:
            return (lam*c_u_w/c_u_dot) + (1 - lam)*(self.prob_dict[self.ngrams-1][considered_text])
        else:
            return (1 - lam)*(self.prob_dict[self.ngrams-1][considered_text])

    def probs(self):
        """Returns a dict mapping from all characters in the vocabulary to the
probabilities of each character."""
        #compute probs of unknown ngrams
        #for i,ngram_dict in self.grams.items():
        #   ngram_dict['unk'] = list(ngram_dict.values()).count(1.) / sum(ngram_dict.values())
        ret_dict = {}
        for i in range(0,self.ngrams):
            ret_dict[i+1] = {}
        self.smooth_one_gram(ret_dict)
        self.witten_bell_smooth(ret_dict)
        self.prob_dict = ret_dict
        return ret_dict

    def smooth_one_gram(self,ret_dict):
        """Returns smoothed 1-gram model"""
        char_count = sum(self.grams[1].values())
        vocab_len = len(self.grams[1].keys())
        lambda_char = char_count / (char_count + vocab_len)
        for char, count in self.grams[1].items():
            
            char = ''.join(char)
            ret_dict[1][char] = (lambda_char * count / char_count) + (1 - lambda_char)*(1./vocab_len) 
        return

    
    def witten_bell_smooth(self,ret_dict):

        # start with 2-grams and continue upwards
        # compute lambda
        for dict_num, count_dict in self.grams.items():
            
            if dict_num != 1.:
                
                for gram, count in count_dict.items():
                    # get lambda value - input is all chars besides last char of ngram
                    c_u_dot,lam = self.lambda_compute(gram[0:len(gram)-1], dict_num)
                    c_u_w = count
                    recurse_p_wu = ret_dict[dict_num - 1][''.join(gram[1:])]
                    ret_dict[dict_num][''.join(gram)] = (lam*c_u_w/c_u_dot) + (1.-lam)*recurse_p_wu



class Application(tk.Frame):
    def __init__(self, model, master=None):
        self.model = model

        tk.Frame.__init__(self, master)
        self.pack()

        self.INPUT = tk.Text(self)
        self.INPUT.pack()

        self.chars = ['qwertyuiop',
                      'asdfghjkl',
                      'zxcvbnm,.',
                      ' ']

        self.KEYS = tk.Frame(self)
        for row in self.chars:
            r = tk.Frame(self.KEYS)
            for w in row:
                # trick to make button sized in pixels
                f = tk.Frame(r, height=32)
                b = tk.Button(f, text=w, command=lambda w=w: self.press(w))
                b.pack(fill=tk.BOTH, expand=1)
                f.pack(side=tk.LEFT)
                f.pack_propagate(False)
            r.pack()
        self.KEYS.pack()

        self.TOOLBAR = tk.Frame()

        self.BEST = tk.Button(self.TOOLBAR, text='Best', command=self.best, 
                              repeatdelay=500, repeatinterval=1)
        self.BEST.pack(side=tk.LEFT)

        self.WORST = tk.Button(self.TOOLBAR, text='Worst', command=self.worst, 
                               repeatdelay=500, repeatinterval=1)
        self.WORST.pack(side=tk.LEFT)

        self.RANDOM = tk.Button(self.TOOLBAR, text='Random', command=self.random, 
                                repeatdelay=500, repeatinterval=1)
        self.RANDOM.pack(side=tk.LEFT)

        self.QUIT = tk.Button(self.TOOLBAR, text='Quit', command=self.quit)
        self.QUIT.pack(side=tk.LEFT)

        self.TOOLBAR.pack()

        self.update()
        self.resize_keys()

    def resize_keys(self):
        for bs, ws in zip(self.KEYS.winfo_children(), self.chars):
            wds = [150*self.model.prob(w)+15 for w in ws]
            wds = [int(wd+0.5) for wd in wds]

            for b, wd in zip(bs.winfo_children(), wds):
                b.config(width=wd)

    def press(self, w):
        self.INPUT.insert(tk.END, w)
        self.INPUT.see(tk.END)
        self.model.read(w)
        self.resize_keys()

    def best(self):
    
        _, w = max((p, w) for (w, p) in self.model.probs()[self.model.ngrams].items())
        self.press(w)

    def worst(self):
        _, w = min((p, w) for (w, p) in self.model.probs()[self.model.ngrams].items())
        self.press(w)

    def random(self):
        s = 0.
        r = random.random()
        p = self.model.probs()
        for w in p:
            s += p[self.model.ngrams][w]
            if s > r:
                break
        self.press(w)

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
        m.compute_test(args.test)   

    except ValueError:
        print("Error: Invalid ngram argument. Please enter an integer argument")
        

    #root = tk.Tk()
    #app = Application(m, master=root)
    #app.mainloop()
    #root.destroy()