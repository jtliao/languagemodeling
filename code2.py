import csv
import os
import nltk
import nltk.data
import math

import code


sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")


def calc_unigram_perplexity(unigram_probs, tokens):
    token_length = 0
    
    summation = 0
    for token in tokens:
        word = token.lower()
        
        if (word == "<" or word == ">" or word == "|" or word == "#" or
            word == "'" or word == '"' or word == '`' or word == '``' or 
            word == "@" or "." in word or word == "(" or word == ")"):
                continue
        token_length += 1
        # Fix when we get smoothing to work
        if word in unigram_probs:
            prob_of_token = unigram_probs[word]
        else:
            prob_of_token = unigram_probs["unk"]
                    
#         print("unigram " + str(prob_of_token))
#         print(word + " " + str(prob_of_token))
#         print(-math.log(prob_of_token))

        summation += (-math.log(prob_of_token))
#         print(summation / token_length)
                
#     print(1/token_length * summation) 
    pp = math.exp(1/token_length * summation)
#     print(pp)
    return pp


def calc_bigram_perplexity(bigram_probs, unknown_bigram_prob, tokens):
    token_length = 0
    
    # Use bigram probabilities to calculate perplexity
    prev_word = None
    summation = 0
    
    for token in tokens:
        word = token.lower()
        
        if (word == "<" or word == ">" or word == "|" or word == "#" or
            word == "'" or word == '"' or word == '`' or word == '``' or 
            word == "@" or "." in word or word == "(" or word == ")"):
                continue
            
        token_length += 1
        
        # Don't do any calc for first -s- (no previous word)
        if prev_word is not None:
            pair = (prev_word, word)
            # Fix when we get smoothing to work
            if pair in bigram_probs:
                prob_of_pair = bigram_probs[pair]
                print("bigram " + str(prob_of_pair))
            else:
#                 prob_of_pair = .1
                prob_of_pair = unknown_bigram_prob
                print("unk bigram " + str(prob_of_pair))
#             print(prob_of_pair)
    #         print("Bigram " + str(prob_of_pair))
            summation += (-math.log(prob_of_pair))
#             print(summation / token_length)
                
        prev_word = word
    pp = math.exp(1/token_length * summation)
#     print(pp)
    return pp


def calc_trigram_perplexity(trigram_probs, unknown_trigram_prob, tokens):
    token_length = 0
    
    # Use trigram probabilities to calculate perplexity
    two_prev_word = None
    prev_word = None
    summation = 0
    
    for token in tokens:
        word = token.lower()
        
        if (word == "<" or word == ">" or word == "|" or word == "#" or
            word == "'" or word == '"' or word == '`' or word == '``' or 
            word == "@" or "." in word or word == "(" or word == ")"):
                continue
            
        token_length += 1
        
        # Don't do any calc for first -s- (no previous word)
        if two_prev_word is not None and prev_word is not None:
            triple = (two_prev_word, prev_word, word)
            # Fix when we get smoothing to work
            if triple in trigram_probs:
                prob_of_triple = trigram_probs[triple]
                print("triple " + str(prob_of_triple))
            else:
#                 prob_of_pair = .1
                prob_of_triple = unknown_trigram_prob
                print("unk triple " + str(prob_of_triple))
                
            summation += (-math.log(prob_of_triple))
#             print(summation / token_length)
        
        two_prev_word = prev_word        
        prev_word = word
        
#     print(unknown_bigram_prob)
#     print("unknown " + str(unknown_count))
#     print("known " + str(known_count))
    pp = math.exp(1/token_length * summation)
#     print(pp)
    return pp

# topics_dir in our case is "data_corrected/classification task"
def calc_all_perplexities(topics_dir):
    perplexity_table = {}
    
    topic_to_ngram_dict = get_topic_to_ngram_dict(topics_dir)
    
    test_dir = os.path.join(topics_dir, "test_for_classification")
    for filename in os.listdir(test_dir):
        with open(os.path.join(test_dir, filename), "r") as f:
            test_text = f.readline()
            
            sentence_list = sentence_detector.tokenize(test_text)

            # Add sentence boundary tags to each sentence
            added_sentence_tags_list = ["-s- " + sentence + " -/s-" for sentence in sentence_list]
#             print(added_sentence_tags_list)
            
            # Tokenize the sentences by words
            tokens = nltk.word_tokenize(" ".join(added_sentence_tags_list))
                        
            for topic, (unigram_probs, bigram_probs, trigram_probs, zero_prob_bigram, zero_prob_trigram) in topic_to_ngram_dict.items():
                unigram_pp = calc_unigram_perplexity(unigram_probs, tokens)
                bigram_pp = calc_bigram_perplexity(bigram_probs, zero_prob_bigram, tokens)
                trigram_pp = calc_trigram_perplexity(trigram_probs, zero_prob_trigram, tokens)
                if filename not in perplexity_table:
                    perplexity_table[filename] = {}
                perplexity_table[filename][topic] = (unigram_pp, bigram_pp, trigram_pp)
    return perplexity_table
                
# Simple classification is just computing perplexity for topic
# and seeing which topic gives the lowest perplexity                
def predict_topic(topics_dir, pred_filename, topic_to_ngram_dict, smoothing_param=3):
    #set to max int value   
    min_unigram_perplexity = 1000000000
    best_unigram_topic_guess = ""
    
    min_bigram_perplexity = 1000000000
    best_bigram_topic_guess = ""
    
    min_trigram_perplexity = 1000000000
    best_trigram_topic_guess = ""
    
    with open(pred_filename, "r") as f:
        text = f.readline()
        
        sentence_list = sentence_detector.tokenize(text)

        # Add sentence boundary tags to each sentence
        added_sentence_tags_list = ["-s- " + sentence + " -/s-" for sentence in sentence_list]
#             print(added_sentence_tags_list)
            
        # Tokenize the sentences by words
        tokens = nltk.word_tokenize(" ".join(added_sentence_tags_list))
        
        for topic, (unigram_probs, bigram_probs, trigram_probs, zero_prob_bigram, zero_prob_trigram) in topic_to_ngram_dict.items():
                
            pp_unigram = calc_unigram_perplexity(unigram_probs, tokens)
            if pp_unigram < min_unigram_perplexity:
                min_unigram_perplexity = pp_unigram
                best_unigram_topic_guess = topic
                
            pp_bigram = calc_bigram_perplexity(bigram_probs, zero_prob_bigram, tokens)
            if pp_bigram < min_bigram_perplexity:
                min_bigram_perplexity = pp_bigram
                best_bigram_topic_guess = topic
             
                
            pp_trigram = calc_trigram_perplexity(trigram_probs, zero_prob_trigram, tokens)
            if pp_trigram < min_trigram_perplexity:
                min_trigram_perplexity = pp_trigram
                best_trigram_topic_guess = topic
                
    return (best_unigram_topic_guess, best_bigram_topic_guess, best_trigram_topic_guess)

    
def get_topic_to_ngram_dict(topics_dir, smoothing_param=3):
    topic_to_ngram_dict = {}
    for topic in os.listdir(topics_dir):
        # Get all topic directories besides test dir
        if topic == "test_for_classification":
            continue
        [unigram_probs, bigram_probs, trigram_probs, zero_prob_bigram, zero_prob_trigram] = code.find_ngram_prob(
            os.path.join(topics_dir, topic, "train_docs"), smoothing_param)     
        topic_to_ngram_dict[topic] = (unigram_probs, bigram_probs, trigram_probs, zero_prob_bigram, zero_prob_trigram)
    return topic_to_ngram_dict    
         

# I split them manually instead (first 61 in validation, rest in training)
# def split_training_and_validation(topics_dir):
#     training_file_to_topic_dict = {}
#     validation_file_to_topic_dict = {}
#     
#     for topic in os.listdir(topics_dir):
#         training_files = os.path.join(topics_dir, topic, "train_docs")
#         num_files = len(training_files)
#         
#         # Design decision to do a 80-20 split
#         for i in range(0, num_files * 0.8):
#             filename = training_files[i]
#             training_file_to_topic_dict[filename] = topic 
#         
#         for i in range(num_files * 0.8, num_files):
#             filename = training_files[i]
#             validation_file_to_topic_dict[filename] = topic
#         
#     return training_file_to_topic_dict, validation_file_to_topic_dict

def classify_topics_validation(topics_dir):
    best_accuracy_unigram = 0
    best_accuracy_bigram = 0
    
    best_param_unigram = 0
    best_param_bigram = 0
    
    # Tune...
    for i in range(3,11):
        smoothing_param = i
        
        topic_to_ngram_dict = get_topic_to_ngram_dict(topics_dir, smoothing_param)
            
        # For computing accuracy
        num_correct_unigram = 0
        num_correct_bigram = 0
        
        num_total = 0
        
        for topic in os.listdir(topics_dir):
            if topic == "test_for_classification":
                continue
            
            for validation_file in os.listdir(
                os.path.join(topics_dir, topic, "validation_docs")):
                
                validation_file = os.path.join(topics_dir, topic, "validation_docs", validation_file)
                
                (best_unigram_topic_guess, best_bigram_topic_guess) = predict_topic(topics_dir, validation_file, topic_to_ngram_dict, smoothing_param)
#                 print((best_unigram_topic_guess, best_bigram_topic_guess))
                
                if topic == best_unigram_topic_guess:
                    num_correct_unigram += 1
                    
                if topic == best_bigram_topic_guess:
                    num_correct_bigram += 1
                    
                num_total += 1
                
        accuracy_unigram = num_correct_unigram / num_total
        accuracy_bigram = num_correct_bigram / num_total
        
        print("unigram " + str(i) + " accuracy is " + str(accuracy_unigram))
        print("bigram " + str(i) + " accuracy is " + str(accuracy_bigram))
        
        if accuracy_unigram > best_accuracy_unigram:
            best_accuracy_unigram = accuracy_unigram
            best_param_unigram = smoothing_param
            
        if accuracy_bigram > best_accuracy_bigram:
            best_accuracy_bigram = accuracy_bigram
            best_param_bigram = smoothing_param
            
    return best_accuracy_unigram, best_param_unigram, best_accuracy_bigram, best_param_bigram



def classify_topics_test(topics_dir, smoothing_param):
    topic_to_ngram_dict = get_topic_to_ngram_dict(topics_dir, smoothing_param)
    
    topic_to_num = {"atheism": 0,
                    "autos": 1,
                    # I use graphics instead of computer_graphics because that's what the dir_name is
                    "graphics": 2,
                    "medicine": 3,
                    "motorcycles": 4,
                    "religion": 5,
                    "space": 6}
    
    test_dir = os.path.join(topics_dir, "test_for_classification")
    with open("test_topics.csv", "w") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Id", "Prediction"])
        for file in os.listdir(test_dir):
            test_file = os.path.join(test_dir, file)
            
            (_, best_bigram_topic_guess) = predict_topic(topics_dir, test_file, topic_to_ngram_dict, smoothing_param)
            csv_writer.writerow([file, topic_to_num[best_bigram_topic_guess]])


                
def main():
#     calc_all_perplexities("data_corrected/classification task")
    print(calc_all_perplexities("data_corrected/classification task"))

#     print(predict_topic("data_corrected/classification task", "data_corrected/classification task/test_for_classification/file_186.txt"))

#     topic_to_ngram_dict = get_topic_to_ngram_dict("data_corrected/classification task", 3)    
#     print(classify_topics_validation("data_corrected/classification task"))
#     classify_topics_test("data_corrected/classification task", 3)

if __name__ == '__main__':
    main()
            
    