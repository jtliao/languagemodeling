import nltk
import numpy as np

#data_corrected\classification task\atheism\train_docs\atheism_file1.txt


def find_ngram_prob(n, tokens):
    ln = len(tokens)
    a = {}
    for i in range(0, ln - n + 1):
        if n == 1:
            token_list = tokens[i]
        else:
            token_list = ()
            for j in range(i, i+n):
                token_list = token_list + (tokens[j],)
        if token_list in a:
            a[token_list] += (1./ln)
        else:
            a[token_list] = (1./ln)
    return a

#remove used words? how to deal with multiple punctuation in a row?
def rand_sentence(prob_table, n):
    match = '-s-'
    sentence = '-s-'
    is_punct = True
    punct = ['.', ',', '?', '!']
    while match != '-/s-':
        ngram = []
        prob = []
        if n == 1:
            ngram = prob_table.keys()
            prob = prob_table.values()
            match = np.random.choice(ngram, 1, prob)[0]
            match = match.astype("string")
            while (match == '-s-') | (is_punct & ((match == '.') |
                  (match == ',') | (match == '!') | (match == '?'))):
                match = np.random.choice(ngram, 1, prob)[0]
        if n == 2:
            for k in prob_table.keys():
                if k[0] == match:
                    ngram.append(k)
                    prob.append(prob_table.get(k))
            match = ngram[np.random.choice(len(ngram), 1, prob)[0]][1]
        if match in punct:
            is_punct = True
            sentence += match
        else:
            is_punct = False
            sentence += " "+match
    return sentence

if __name__ == '__main__':
    n = int(raw_input("Enter value of n \n"))
    text = ""
    while True:
        data = raw_input("Enter file, or enter blank if no more files: \n")
        if not data.lower() == "":
            with open(data, 'r') as my_file:
                text += my_file.read().replace('\n', '')
            continue
        else:
            break
    token = nltk.word_tokenize(text)
    ng = find_ngram_prob(n, token)
    #print ng
    print rand_sentence(ng, n)
