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
            self.prob_dict[i+1] = {}
    def train(self, filename):
        """Train the model on a text file."""

        for line in open(filename):
            line.replace('\n',' ')
            line = '`'*(self.ngrams-1) + line
            # form all ngrams from line
            for i in range(1,self.ngrams+1):

                ngrams_list = nltk.ngrams(line, i)
                # add to ngram dictionary
                for entry in ngrams_list:
                    if len(entry) == 1:
                        context = ''
                    else:
                        context = ''.join(entry[0:len(entry)-1])
                    letter = ''.join(entry[len(entry)-1])

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
        ten_pos.replace('\n',' ')
        # append start symbols handled in self.start()
        self.compute(ten_pos)
        return
    def compute_test(self,filename):

        test_set = open(filename)

        f_str = test_set.read()
        f_str.replace('\n',' ')
        self.compute(f_str)
        return
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
        return
    def predict(self):
        """ Returns char with highest probability given current state """
        #print("state<" + self.state + ">")

        max_prob = 0.
        ret_pair = ['',0.]
        if self.ngrams == 1:
            context = "" 
        else:
            context = self.state[(len(self.state))-(self.ngrams-1):]
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
                c_u_w = 0
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
            lam = c_u_dot / (c_u_dot + n_plus_u)
        return c_u_dot, lam
        


    # The following two methods make the model work like a finite
    # automaton.
    def start(self):
        """Resets the state."""
        self.state = "`"*(self.ngrams-1)
        return

    def read(self, w):
        """Reads in character w, updating the state."""
        self.state = self.state + w
        return

    # The following two methods add probabilities to the finite automaton.





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
