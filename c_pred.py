import random
import tkinter as tk
import nltk as nltk
class Uniform(object):
    """Barebones example of a language model class."""

    def __init__(self,ngrams):
        self.grams =     {}
        self.c_map =     {}
        self.ngrams =    ngrams
        self.prob_dict = {}
        self.state = []
        self.char_lookup={}
        self.shared_chars = "maeon"
        for i in range(0,ngrams):  
            self.grams[i+1]     = {}
            self.prob_dict[i+1] = {}

    def read_char_map(self, f_map):
        i = 0
        for line in open(f_map):
            line = line.split()
            char = line[0]
            reading = line[1]
            self.c_map[char] = reading
        for char in self.shared_chars:
            self.c_map[char] = char
        self.c_map['<space>'] = ' '
        print("There are " + str(list(self.c_map.values()).count('yi')) + " characters read as yi")

    def train(self, filename):
        """Train the model on a text file."""
        i_file = open(filename)
        train_set = i_file.read()
        train_set = train_set.replace('\n', '')


        # form all ngrams from line

        for i in range(1,self.ngrams+1):
                        
            ngrams_list = nltk.ngrams('σ'*(i-1) + train_set, i)
            # add to ngram dictionary
            for entry in ngrams_list:
                if len(entry) == 1:
                    context = ''
                else:
                    context = ''.join(entry[:-1])
                letter = entry[-1]
                if context in self.grams[i].keys():
                   # if char has been seen in this context

                   if letter in self.grams[i][context].keys():                      
                        self.grams[i][context][letter] += 1.
                   else:   
                        self.grams[i][context][letter] = 1
                else:
                   self.grams[i][context] = {}                      
                   self.grams[i][context][letter] = 1
                   # add to dictionary / build vocabulary

        return
    def compute_dev(self, filename, correct_file):
        """ Read in a file, predict most probably char for each char position in file """

        dev_set = open(filename)
        correct_set = open(correct_file)
        
        ten_pos = dev_set.read(50)
        ten_pos = ten_pos.replace('\n',' ')  
        ten_pos = ten_pos.split()
        ten_pos = ten_pos[0:10]
        c_set = correct_set.read()
        c_set = c_set.replace('\n',' ')

        tokenized_set = []
        for i,value in enumerate(c_set):
            tokenized_set.append(value)
        
        # append start symbols handled in self.start()
        output = self.compute(ten_pos,prin='t')
        self.compare(output, tokenized_set, "dev set")
        return
    def compute_test(self,filename,c_file):
        self.start()
        test_set = open(filename)
        correct_set = open(c_file)

        f_str = test_set.read()
        f_str = f_str.replace('\n', ' ')
        f_split = f_str.split()
        
        c_set = correct_set.read()
        c_set = c_set.replace('\n','')
        #TODO Make sets line up after newlines
        tokenized_set = []
        for i,value in enumerate(c_set):
            tokenized_set.append(value)
        output = self.compute(f_split,prin='f')
        self.compare(output,tokenized_set, "test set")
        return
    def compare(self, output, correct_tokens, name):
        correct = 0
        incorrect = 0

        for i,value in enumerate(output):

            
            if value == correct_tokens[i]:
                correct += 1
            else:
                print("guess<" + value + ">, correct<" + correct_tokens[i] + "> ")
                incorrect += 1
        print("total accuracy on " + name + " = " + str( correct / (correct+incorrect) ) )

    def compute(self,f_string,prin='f'):
        correct = 0
        incorrect = 0
        ret_string = ""
        f_string = list(f_string)
        for i,char in enumerate(f_string):
            self.read(char)
            guess,prob = self.predict()
            if prin == 't':
                print('pos< ' + str(i) + ' >, guess< ' + guess + ' >, prob< ' + str(prob) + ' >')
            ret_string = ret_string + guess

        return ret_string
    def predict(self):
        """ Returns char with highest probability given current state """
        #print("state<" + self.state + ">")

        max_prob = 0.
        ret_pair = ['',0.]
        reading = self.state[-1] 

        # indiviual chars map to chars, <space> maps to ' ' w/ 100% accuracy
        if reading == '<space>':
            return ' ',1
        if self.state[-1] not in self.c_map.values():
            return self.state[-1], 1
        #if reading in self.chars:
      
         #   return self.c_map[self.state[len(self.state)-1]],1


        if self.ngrams == 1:
            context = "" 
        else:
            context = self.state[-(self.ngrams): -1]

        for key, value in self.c_map.items():
        
            
            if value == reading:
                prob = self.recurse_smooth(key, context, self.ngrams)    
                if prob > max_prob:
                    max_prob = prob
                    if key == '<space>':
                        ret_pair = [' ',prob]
                    else:                     
                        ret_pair = [key,prob]
        return ret_pair[0],ret_pair[1]

    def recurse_smooth(self,symbol, context,dict_num):
        """ Computes recursively p(w | u) """
        lookup = self.c_map[symbol]
        context = ''.join(context)
        
        if context in self.prob_dict[dict_num].keys():
            if symbol in self.prob_dict[dict_num][context].keys():

                return self.prob_dict[dict_num][context][symbol]
            else:
                self.prob_dict[dict_num][context] = {}
     
        else:
            self.prob_dict[dict_num][context] = {}

        # unigram
        if dict_num == 1:
            self.prob_dict[dict_num][context][symbol] = self.unigram_prob(symbol) 
            return self.prob_dict[dict_num][context][symbol]


        c_u_dot,lam = self.lambda_compute(context, dict_num)
        c_u_w = 0
        if context in self.grams[dict_num].keys():
            if symbol in self.grams[dict_num][context].keys():
                c_u_w = self.grams[dict_num][context][symbol]
               
     
        if len(context) == 1:
            new_context = ""
        else:
            new_context = context[1:]
        if c_u_dot == 0:
                self.prob_dict[dict_num][context][symbol] =(0 + (1.-lam)*self.recurse_smooth(symbol,new_context,dict_num-1))   
                return self.prob_dict[dict_num][context][symbol] 
        else: 
                self.prob_dict[dict_num][context][symbol] =((lam*c_u_w/c_u_dot) + (1.-lam)*self.recurse_smooth(symbol,new_context,dict_num-1))
                return self.prob_dict[dict_num][context][symbol]


    def unigram_prob(self,symbol):
        """ Returns unigram prob of one single char in model """
        lookup = self.c_map[symbol] 

        char_count = vocab_len = lambda_char = count = 0
        vocab_len = len(self.grams[1][""].keys())
        char_count = sum(self.grams[1][""].values())

        if (char_count+vocab_len) == 0:
            lambda_char =0
        else:
            lambda_char = char_count / (char_count + vocab_len)

        if symbol in self.grams[1][""].keys():
            count = self.grams[1][""][symbol]
        else:
            count = 0
        if char_count == 0:
            return (0 + (1-lambda_char)*(1./vocab_len))
        else:
            return ((lambda_char * count / char_count) + (1-lambda_char)*(1./vocab_len))

    def lambda_compute(self, context, dict_num):
        """ Computes lambda value necessary for witten bell smoothing """
        # compute c(u o )
        c_u_dot = 0
        n_plus_u = 0
        if context in self.grams[dict_num].keys():
            c_u_dot += sum(self.grams[dict_num][context].values())
            n_plus_u += len(self.grams[dict_num][context].keys())

        if (c_u_dot + n_plus_u) == 0:
            lam = 0
        else:
            lam = c_u_dot / (c_u_dot + n_plus_u+1)
        return c_u_dot, lam
        


    # The following two methods make the model work like a finite
    # automaton.
    def start(self):
        """Resets the state."""
        state = 'σ'*(self.ngrams-1)
        self.state = list(state)
        return

    def read(self, w):
        """Reads in character w, updating the state."""
        self.state.append(w)
        return

    # The following two methods add probabilities to the finite automaton.





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='map')
    parser.add_argument(dest='train')
    parser.add_argument(dest='dev')

    parser.add_argument(dest='cor_dev')
    parser.add_argument(dest='test')
    parser.add_argument(dest='cor_test')
    parser.add_argument(dest='ngrams')
    args = parser.parse_args()

    ##### Replace this line with an instantiation of your model #####
    #try:
    
    ngrams = int(args.ngrams)
    m = Uniform(ngrams)
    m.read_char_map(args.map)
    m.train(args.train)
    m.start()
    m.compute_dev(args.dev,args.cor_dev)
    m.compute_test(args.test,args.cor_test)   

    #except ValueError:
     #   print("Error: Invalid ngram argument. Please enter an integer argument")
        

    #root = tk.Tk()
    #app = Application(m, master=root)
    #app.mainloop()
    #root.destroy()
