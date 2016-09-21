import os
import nltk
import nltk.data
import math

import code


sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")


def calc_bigram_perplexity(bigram_probs, unknown_bigram_prob, tokens):
    token_length = len(tokens)
    
    # Use bigram probabilities to calculate perplexity
    prev_token = "-s-"
    summation = 0
    
    unknown_count = 0
    known_count = 0
    
    for token in tokens:
        token = token.lower()
                    
        pair = (prev_token, token)
        # Fix when we get smoothing to work
        if pair in bigram_probs:
            prob_of_pair = bigram_probs[pair]
#             print("seen " + str(prob_of_pair))
            unknown_count += 1
        else:
            prob_of_pair = .1
#             print("unseen " + str(unknown_bigram_prob))
            known_count += 1
                 
#         print("Bigram " + str(prob_of_pair))
        summation += (-math.log(prob_of_pair))
                
        prev_token = token
#     print("unknown " + str(unknown_count))
#     print("known " + str(known_count))
    pp = math.exp(1/token_length * summation)
#     print(pp)
    return pp


def calc_unigram_perplexity(unigram_probs, tokens):
    token_length = len(tokens)
    
    # Use bigram probabilities to calculate perplexity
    
    summation = 0
    for token in tokens:
        token = token.lower()
                    
        # Fix when we get smoothing to work
        if token in unigram_probs:
            prob_of_token = unigram_probs[token]
        else:
            prob_of_token = unigram_probs["unk"]
                    
#         print("unigram " + str(prob_of_token))
        summation += (-math.log(prob_of_token))
                
                
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
                        
            for topic, (unigram_probs, bigram_probs, unknown_bigram_prob) in topic_to_ngram_dict.items():
                unigram_pp = calc_unigram_perplexity(unigram_probs, tokens)
                bigram_pp = calc_bigram_perplexity(bigram_probs, unknown_bigram_prob, tokens)
                if filename not in perplexity_table:
                    perplexity_table[filename] = {}
                perplexity_table[filename][topic] = (unigram_pp, bigram_pp)
    return perplexity_table
                
# Simple classification is just computing perplexity for topic
# and seeing which topic gives the lowest perplexity                
def predict_topic(topics_dir, pred_filename, topic_to_ngram_dict, smoothing_param=3):
    #set to max int value
    min_bigram_perplexity = 1000000
    best_bigram_topic_guess = ""
    
    min_unigram_perplexity = 1000000
    best_unigram_topic_guess = ""
    
    with open(pred_filename, "r") as f:
        text = f.readline()
        
        sentence_list = sentence_detector.tokenize(text)

        # Add sentence boundary tags to each sentence
        added_sentence_tags_list = ["-s- " + sentence + " -/s-" for sentence in sentence_list]
#             print(added_sentence_tags_list)
            
        # Tokenize the sentences by words
        tokens = nltk.word_tokenize(" ".join(added_sentence_tags_list))
        
        for topic, (unigram_probs, bigram_probs, unknown_bigram_prob) in topic_to_ngram_dict.items():
            pp_unigram = calc_bigram_perplexity(bigram_probs, unknown_bigram_prob, tokens)
            if pp_unigram < min_bigram_perplexity:
                min_bigram_perplexity = pp_unigram
                best_bigram_topic_guess = topic
                
            pp_bigram = calc_unigram_perplexity(unigram_probs, tokens)
            if pp_bigram < min_unigram_perplexity:
                min_unigram_perplexity = pp_bigram
                best_unigram_topic_guess = topic 
                
    return (best_unigram_topic_guess, best_bigram_topic_guess)

    
def get_topic_to_ngram_dict(topics_dir, smoothing_param=3):
    topic_to_ngram_dict = {}
    for topic in os.listdir(topics_dir):
        # Get all topic directories besides test dir
        if topic == "test_for_classification":
            continue
        [unigram_probs, bigram_probs, zero_prob_bigram] = code.find_ngram_prob(
            os.path.join(topics_dir, topic, "train_docs"), smoothing_param)     
        topic_to_ngram_dict[topic] = (unigram_probs, bigram_probs, zero_prob_bigram)
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

def classify_topics(topics_dir, topic_to_ngram_dict):
    best_accuracy_unigram = 0
    best_accuracy_bigram = 0
    
    #TODO: Change this to actual tuning param
    best_param_unigram = 0
    best_param_bigram = 0
    
    # Tune...
    for i in range(3,11):
        smoothing_param = i
        #TODO -- actual hyperparameter tuning
        
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
                print((best_unigram_topic_guess, best_bigram_topic_guess))
                
                if topic == best_unigram_topic_guess:
                    num_correct_unigram += 1
                    
                if topic == best_bigram_topic_guess:
                    num_correct_bigram += 1
                    
                num_total += 1
                
        accuracy_unigram = num_correct_unigram / num_total
        accuracy_bigram = num_correct_bigram / num_total
        
        if accuracy_unigram > best_accuracy_unigram:
            best_accuracy_unigram = accuracy_unigram
            best_param_unigram = smoothing_param
            
        if accuracy_bigram > best_accuracy_bigram:
            best_accuracy_bigram = accuracy_bigram
            best_param_bigram = smoothing_param
            
    return best_accuracy_unigram, best_param_unigram, best_accuracy_bigram, best_param_bigram
                
def main():
    
#     topic_to_ngram_dict = get_topic_to_ngram_dict("data_corrected/classification task", 3)
    
    print(calc_all_perplexities("data_corrected/classification task"))
    
#     print(calc_all_perplexities("data_corrected/classification task"))
#     print(predict_topic("data_corrected/classification task", "data_corrected/classification task/test_for_classification/file_186.txt"))
#     print(classify_topics("data_corrected/classification task", topic_to_ngram_dict))

if __name__ == '__main__':
    main()
            
    