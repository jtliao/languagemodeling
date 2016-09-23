import nltk
import nltk.data
import numpy as np
import os


def find_ngram_counts(dirname):
    sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")
    
    unigram_counts = {}
    bigram_counts = {}
    trigram_counts = {}
    
    unk_words = set()
    all_tokens = []
    for filename in os.listdir(dirname):
        with open(os.path.join(dirname,filename), 'r') as f:
            text = f.readline()
            
            sentence_list = sentence_detector.tokenize(text)

            # Add sentence boundary tags to each sentence
            added_sentence_tags_list = ["-s- " + sentence + " -/s-" for sentence in sentence_list]
#             print(added_sentence_tags_list)
            
            # Tokenize the sentences by words
            tokens = nltk.word_tokenize(" ".join(added_sentence_tags_list))
            
#             unk_words = set()
            for word in tokens:
                
                
                # Design decision to ignore these special characters and . inside of words (websites/email addresses)
                if (word == "<" or word == ">" or word == "|" or word == "#" or
                    word == "'" or word == '"' or word == '`' or word == '``' or 
                    word == "@" or "." in word or word == "(" or word == ")"):
                    continue

                # Design decision to ignore case
                word = word.lower()
                
                all_tokens.append(word)

                if word not in unigram_counts:
#                     if word not in unk_words:
#                         unk_words.add(word)
#                         word = 'unk'
#                         if 'unk' not in unigram_counts:
#                             unigram_counts['unk'] = 1
#                         else:
#                             unigram_counts['unk'] += 1
#                     else:
#                         unigram_counts[word] = 1
#                         unk_words.remove(word)
                    unigram_counts[word] = 1
                else:
                    unigram_counts[word] += 1

    for unigram, count in unigram_counts.items():
        if count == 1:
            word = unigram
            unk_words.add(word)
    if len(unk_words) != 0:
        for word in unk_words:
            del unigram_counts[word]
        unigram_counts["unk"] = len(unk_words)
        
        
#     for filename in os.listdir(dirname):
#         with open(os.path.join(dirname,filename), 'r') as f:   
#             text = f.readline()
#             
#             sentence_list = sentence_detector.tokenize(text)
# 
#             # Add sentence boundary tags to each sentence
#             added_sentence_tags_list = ["-s- " + sentence + " -/s-" for sentence in sentence_list]
# #             print(added_sentence_tags_list)
#             
#             # Tokenize the sentences by words
#             tokens = nltk.word_tokenize(" ".join(added_sentence_tags_list))

    prev_word = None
    prev2_word = None
    for word in all_tokens:
        if word in unk_words:
            word = "unk"
        # Cannot compute bigram for first pass
        if prev_word is not None:
            pair = (prev_word, word)
            if pair not in bigram_counts:
                bigram_counts[pair] = 1
            else:
                bigram_counts[pair] += 1
            if prev2_word is not None:
                tup = (prev2_word, prev_word, word)
                if tup not in trigram_counts:
                    trigram_counts[tup] = 1
                else:
                    trigram_counts[tup] += 1
        prev2_word = prev_word
        prev_word = word

    return unigram_counts, bigram_counts, trigram_counts


def find_ngram_prob(dirname, smoothing_param=3):
    unigram_counts, bigram_counts, trigram_counts = find_ngram_counts(dirname)
    num_word_types = len(unigram_counts)
    unigram_counts = smooth(unigram_counts, 1, num_word_types, smoothing_param)
    bigram_counts, count_zero_bi = smooth(bigram_counts, 2, num_word_types, smoothing_param)
    trigram_counts, count_zero_tri = smooth(trigram_counts, 3, num_word_types, smoothing_param)
    
    
    unigram_probs = {k: v/sum(unigram_counts.values()) for k, v in unigram_counts.items()}
    
    # Bigram probabilities = count(w(n-1) w(n)) / count(w(n-1))
    # The keys in this dict (W1, W2) represent P(W2 | W1)
    bigram_probs = {k: v/unigram_counts[k[0]] for k, v in bigram_counts.items()}

    trigram_probs = {k: v/bigram_counts[(k[0], k[1])] for k, v in trigram_counts.items()}
    
    for bigram, bigram_count in bigram_counts.items():
        trigram_probs[bigram] = count_zero_tri / bigram_count
    trigram_probs["unk"] = count_zero_tri / count_zero_bi
    
    for unigram, unigram_count in unigram_counts.items():
        bigram_probs[unigram] = count_zero_bi / unigram_count
    
    
    return unigram_probs, bigram_probs, trigram_probs, count_zero_bi, count_zero_tri


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
        else:
            for k in prob_table.keys():
                if k[0] == match:
                    ngram.append(k)
                    prob.append(prob_table.get(k))
            prob = np.array(prob) / np.sum(prob)
            if n == 2:
                match = ngram[np.random.choice(len(ngram), 1, p=prob)[0]][1]
            if n == 3:
                match = ngram[np.random.choice(len(ngram), 1, p=prob)[0]][2]
        if (match == '.') or (match == ',') or (match == '!') or (match == '?'):
            is_punct = True
            sentence += match
        else:
            is_punct = False
            sentence += " "+match
    return sentence


def smooth(counts, n, num_word_types, t):
    N = {} #count of counts
    new_count = {} #the adjusted counts for changed counts
    total_count = 0
#     V = len(counts)
    
    #populate N with counts of counts
    for val in counts.values():
#         print(val)
        total_count += val
        if val not in N:
            N[val] = 1
        else:
            N[val] += 1
    #if n=1, c=0 doesn't happen, adjust counts accordingly
    if n == 1:
#         new_count[0] = 0
#         for c in range(1, t):
        for c in range(2, t):
            new_count[c] = (c+1)*N[c+1]/N[c]
    #if n=2, c=0 is the bigrams that have not occurred, adjust counts accordingly
    if n == 2:
        V = num_word_types
        num_bigram_types_seen = len(counts)
        N[0] = (V**2) - num_bigram_types_seen
        
#         print((V**2) - num_bigram_types_seen)
#         print(len(counts)**2 - total_count)
        for c in range(0, t):
            new_count[c] = (c+1)*N[c+1]/N[c]
    if n == 3:
        V = num_word_types
        num_trigram_types_seen = len(counts)
        N[0] = (V**3) - num_trigram_types_seen
        for c in range(0, t):
            new_count[c] = (c+1)*N[c+1]/N[c]
    #traverse through counts, if count has been smoothed, then change the value in counts
    for key, value in counts.items():
        if value in new_count:
            counts[key] = new_count[value]

    if n == 1:
        return counts
#     return counts, new_count[0]/total_count
    return counts, new_count[0]


def main():
#     n = int(input("Enter value of n (only 1 or 2)\n"))
#     text = ""
     
#     data = input("Enter file or directory name of corpus: \n")
    n=2
    data = r"data_corrected\classification task\atheism\train_docs"
     
    unigram_prob, bigram_prob, trigram_prob, _, _ = find_ngram_prob(data)

    print(trigram_prob)
 
    if n == 1:
        prob_table = unigram_prob
    elif n == 2:
        prob_table = bigram_prob
    elif n == 3:
        prob_table = trigram_prob
    else:
        print("Can only do unigram and bigram\n")
        
    unigram_counts, bigram_counts, trigram_counts = find_ngram_counts(data)
#     print(len(bigram_counts.keys()))
#     print(bigram_prob[("unk","unk")])
#     print(unigram_prob)
# #     
#     sum=0
#     num_keys=0
#     for key, val in unigram_counts.items():
#         sum+=val
#         num_keys += 1
#     print(sum)
#     print(num_keys)
    
#     print(prob_table)

#     start_of_sentence = input("Enter partial sentence that you want completed (or leave empty for new sentence) \n")
#     if start_of_sentence == "":
#         start_of_sentence = "-s-"
#     for _ in range(0,5):
#         print(rand_sentence(prob_table, n, start_of_sentence))


if __name__ == '__main__':
    main()
