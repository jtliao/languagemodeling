import nltk
from nltk.util import ngrams


def find_ngram(n, tokens):
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

with open(r'data_corrected\classification task\atheism\train_docs\atheism_file1.txt', 'r') as my_file:
    text = my_file.read().replace('\n', '')
token = nltk.word_tokenize(text)
x = list(ngrams(token, 2))
print find_ngram(2, token)
