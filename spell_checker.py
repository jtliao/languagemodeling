import os
import code
from collections import defaultdict

def spell_checker(dirname, unigram, bigram):
    #use defaultdict to make it easier to use lists as values
    confused = defaultdict(list)
    with open('data_corrected\spell_checking_task\confusion_set.txt', 'r') as f:
        for line in f:
            #add all confused words to dictionary
            splitted = line.split()
            confused[splitted[0]].append(splitted[1])
            confused[splitted[1]].append(splitted[0])
    for filename in os.listdir(dirname+r"\test_modified_docs"):
        output = ''
        with open(os.path.join(dirname+r"\test_modified_docs", filename), 'r') as f:
            text = f.readline()
            words = text.split()
            for ind in range(0, len(words)):
                word = words[ind]
                #first check if in a bigram, since context is more indicative of correct word,
                #then check unigrams
                if word in confused:
                    #find probability of word being in bigram with word before and after
                    bigram_back = bigram.get((words[ind-1], word)) if ind > 0 else None
                    bigram_forward = bigram.get((word, words[ind+1])) if ind < len(words) - 1 else None
                    conf_probs = []
                    #find the highest probability of current word existing
                    highest = 0
                    if bigram_back is not None and bigram_forward is not None:
                        highest = max(bigram_back, bigram_forward)
                    elif bigram_forward is not None:
                        highest = bigram_forward
                    elif bigram_back is not None:
                        highest = bigram_back
                    #find highest probability of a confused word existing
                    for conf in confused[word]:
                        conf_bigram_back = bigram.get((words[ind-1], conf)) if ind > 0 else None
                        conf_bigram_forward = bigram.get((conf, words[ind+1])) if ind < len(words) - 1 else None
                        if conf_bigram_back is not None and conf_bigram_forward is not None:
                            conf_highest = max(conf_bigram_back, conf_bigram_forward)
                        elif conf_bigram_forward is not None:
                            conf_highest = conf_bigram_forward
                        elif conf_bigram_back is not None:
                            conf_highest = conf_bigram_back
                        else:
                            conf_highest = 0
                        conf_probs.append((conf_highest, conf))
                    conf_list = sorted(conf_probs, reverse=True)
                    #if prob is higher for confused word, use it
                    if len(conf_list) > 0:
                        output += (conf_list[0][1] + " ") if highest < conf_list[0][0] else (word + " ")
                        highest = 1
                    else:
                        output += word + " "
                        highest = 1
                    #check prob of current word/confused in unigrams, use highest probability word
                    if highest == 0:
                        if unigram.get(word) is not None:
                            highest = unigram.get(word)
                        for conf in confused[word]:
                            if unigram.get(conf) is not None:
                                conf_probs.append((unigram.get(conf), conf))
                        conf_list = sorted(conf_probs, reverse=True)
                        if len(conf_list) > 0:
                            output += (conf_list[0][1] + " ") if highest < conf_list[0][0] else (word + " ")
                            highest = 1
                        else:
                            output += word + " "
                            highest = 1
                    #if now words were in training set, assume the word is right and just use current word
                    if highest == 0:
                        output += word + " "
                else:
                    output += word + " "
        with open(os.path.join(dirname+r'\test_docs', filename), 'w') as file:
            file.write(output)


def main():
    direc = input("Enter directory name of corpus: \n")
    #data_corrected\spell_checking_task\atheism
    uni, bi, _, _, _ = code.find_ngram_prob(direc + r"\train_docs")
    spell_checker(direc, uni, bi)


if __name__ == '__main__':
    main()
