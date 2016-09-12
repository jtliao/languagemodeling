import nltk
import nltk.data
import numpy as np
import os

from email.parser import Parser

# data_corrected\classification task\atheism\train_docs


def find_ngram_counts(dirname):
    sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")
    
    unigram_counts = {}
    bigram_counts = {}
    
    for filename in os.listdir(dirname):
        with open(os.path.join(dirname,filename), 'r') as f:
            text = f.readline()
            
            sentence_list = sentence_detector.tokenize(text)

            # Add sentence boundary tags to each sentence
            added_sentence_tags_list = ["-s- " + sentence + " -/s-" for sentence in sentence_list]
#             print(added_sentence_tags_list)
            
            # Tokenize the sentences by words
            tokens = nltk.word_tokenize(" ".join(added_sentence_tags_list))
#             print(tokens)
            
            # Alternative: use counts of '(' ')' and '[' ']' but people forget to close their parentheses/brackets...
            is_inside_parens = False
            is_inside_brackets = False
            
            prev_word = None
            
            for word in tokens:
                
                # Design decision to ignore these special characters
                if (word == "<" or word == ">" or word == "|" or word == "#" or
                    word == "'" or word == '"' or word == '`' or word == '``'):
                    continue
                
                # Design decision to ignore every thing that is in parentheses/brackets
                # because this information is by definition superfluous
                elif word == "(":
                    is_inside_parens = True
                    continue
                
                elif word == "[":
                    is_inside_brackets = True
                    continue
                
                elif word == ")":
                    is_inside_parens = False
                    continue
                
                elif word == "]":
                    is_inside_brackets = False
                    continue
                
                if is_inside_parens or is_inside_brackets:
                    continue
                
                # Design decision to ignore case
                word = word.lower()
               
                if word not in unigram_counts:
                    unigram_counts[word] = 1
                else:
                    unigram_counts[word] += 1
                
                # Cannot compute bigram for first pass
                if prev_word is not None:
                    pair = (prev_word, word)
                    if pair not in bigram_counts:
                        bigram_counts[pair] = 1
                    else:
                        bigram_counts[pair] += 1
                    
                prev_word = word
                
            if is_inside_parens or is_inside_brackets:
                print("UNCLOSED PARENS/BRACKETS\n")
                print(filename)
    
#     ln = len(tokens)
#     a = {}
#     for i in range(0, ln - n + 1):
#         if n == 1:
#             token_list = tokens[i].lower()
#         else:
#             token_list = ()
#             for j in range(i, i+n):
#                 token_list = token_list + (tokens[j].lower(),)
#         if token_list in a:
#             a[token_list] += (1./ln)
#         else:
#             a[token_list] = (1./ln)
    return unigram_counts, bigram_counts


def find_ngram_prob(dirname):
    unigram_counts, bigram_counts = find_ngram_counts(dirname)
    
    unigram_probs = {k: v/sum(unigram_counts.values()) for k, v in unigram_counts.items()}
    
    # Bigram probabilities = count(w(n-1) w(n)) / count(w(n-1))
    bigram_probs = {k: v/unigram_counts[k[0]] for k, v in bigram_counts.items()}
    
    return unigram_probs, bigram_probs


#remove used words? how to deal with multiple punctuation in a row?
def rand_sentence(prob_table, n, start_of_sentence='-s-'):
    tokens = nltk.word_tokenize(start_of_sentence)
    
    # set this to the last token in given start_of_sentence
    match = tokens[len(tokens) - 1].lower()
    
    # make sure sentence starts with start sentence token
    if tokens[0] != "-s-":
        sentence = "-s- " + start_of_sentence
    else:
        sentence = start_of_sentence
        
    is_punct = True
    while match != '-/s-':
        ngram = []
        prob = []
        if n == 1:
            ngram = list(prob_table.keys())
            prob = list(prob_table.values())
            match = np.random.choice(ngram, 1, p=prob)[0]
            while (match == '-s-') or (is_punct & ((match == '.') or
                  (match == ',') or (match == '!') or (match == '?'))):
                match = np.random.choice(ngram, 1, p=prob)[0]
        if n == 2:
            for k in prob_table.keys():
                if k[0] == match:
                    ngram.append(k)
                    prob.append(prob_table.get(k))
            match = ngram[np.random.choice(len(ngram), 1, p=prob)[0]][1]
        if (match == '.') or (match == ',') or (match == '!') or (match == '?'):
            is_punct = True
            sentence += match
        else:
            is_punct = False
            sentence += " "+match
    return sentence


def main():
    n = int(input("Enter value of n (only 1 or 2)\n"))
#     text = ""
     
    data = input("Enter file or directory name of corpus: \n")
    
    # data_corrected\classification task\atheism\train_docs
     
    unigram_prob, bigram_prob = find_ngram_prob(data)

    #print(unigram_prob["-/s-"])
 
    if n == 1:
        prob_table = unigram_prob
    elif n == 2:
        prob_table = bigram_prob
    else:
        print("Can only do unigram and bigram\n")
     

    start_of_sentence = input("Enter partial sentence that you want completed (or leave empty for new sentence) \n")
    if start_of_sentence == "":
        start_of_sentence = "-s-"
    for _ in range(0,5):
        print(rand_sentence(prob_table, n, start_of_sentence))


if __name__ == '__main__':
    main()
