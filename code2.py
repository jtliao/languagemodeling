import os
import nltk
import math

import code


def calc_bigram_perplexity(bigram_probs, tokens):
    token_length = len(tokens)
    
    # Use bigram probabilities to calculate perplexity
    prev_token = "-s-"
    summation = 0
    for token in tokens:
        token = token.lower()
                    
        pair = (prev_token, token)
        # Fix when we get smoothing to work
        if pair in bigram_probs:
            prob_of_pair = bigram_probs[pair]
        else:
            prob_of_pair = 0
                    
        summation += (-math.log(prob_of_pair))
                
        prev_token = token
                    
    pp = math.exp(1/token_length * summation)
    print(pp)
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
            prob_of_token = 0
                    
        summation += (-math.log(prob_of_token))
                
                
    pp = math.exp(1/token_length * summation)
    print(pp)
    return pp

# topics_dir in our case is "data_corrected/classification task"
def calc_all_perplexities(topics_dir):
    perplexity_table = {}
    
    test_dir = os.path.join(topics_dir, "test_for_classification")
    for filename in os.listdir(test_dir):
        with open(os.path.join(test_dir, filename), "r") as f:
            test_text = f.readline()
            
            tokens = nltk.word_tokenize(test_text)
            
            topic_to_ngram_dict = get_topic_to_ngram_dict(topics_dir)
            
            for topic, (unigram_probs, bigram_probs) in topic_to_ngram_dict.items():
                unigram_pp = calc_unigram_perplexity(unigram_probs, tokens)
                bigram_pp = calc_bigram_perplexity(bigram_probs, tokens)
                perplexity_table[filename][topic] = (unigram_pp, bigram_pp)
    return perplexity_table
                
# Simple classification is just computing perplexity for topic
# and seeing which topic gives the lowest perplexity                
def predict_topic(topics_dir, pred_filename):
    topic_to_unigram_dict, topic_to_bigram_dict = get_topic_to_ngram_dict(topics_dir)
    #set to max int value
    min_bigram_perplexity = 100000
    best_bigram_topic_guess = ""
    
    with open(pred_filename, "r") as f:
        text = f.readline()
        tokens = nltk.word_tokenize(text)
    
        for topic, bigram_probs in topic_to_bigram_dict.items():
            pp = calc_bigram_perplexity(bigram_probs, tokens)
            if pp < min_bigram_perplexity:
                min_bigram_perplexity = pp
                best_bigram_topic_guess = topic
                
                
    min_unigram_perplexity = 100000
    best_unigram_topic_guess = ""
    
    with open(pred_filename, "r") as f:
        text = f.readline()
        tokens = nltk.word_tokenize(text)
    
        for topic, unigram_probs in topic_to_unigram_dict.items():
            pp = calc_unigram_perplexity(unigram_probs, tokens)
            if pp < min_unigram_perplexity:
                min_unigram_perplexity = pp
                best_unigram_topic_guess = topic
                
                
    return (best_bigram_topic_guess, best_unigram_topic_guess)

    
def get_topic_to_ngram_dict(topics_dir):
    topic_to_ngram_dict = {}
    for topic in os.listdir(topics_dir):
        # Get all topic directories besides test dir
        if topics_dir == "test_for_classification":
            continue
        [unigram_probs, bigram_probs] = code.find_ngram_prob(
            os.path.join(topics_dir, topic, "train_docs"))     
        topic_to_ngram_dict[topic] = (unigram_probs, bigram_probs)
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

def classify_topics(topics_dir):
    best_accuracy = 0
    
    #TODO: Change this to actual tuning param
    best_param = 0
    # Tune...
    for i in range(0,1,0.1):
        smoothing_param = i
        #TODO -- actual hyperparameter tuning
        
        # For computing accuracty
        num_correct = 0
        num_total = 0
        
        for topic in topics_dir:
            for validation_file in os.listdir(
                os.path.join(topics_dir, topic, "validation_docs")):
                
                topic_predicted = predict_topic(topics_dir, validation_file)
                
                if topic == topic_predicted:
                    num_correct += 1
                    
                num_total += 1
                
        accuracy = num_correct / num_total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_param = smoothing_param
            
    return best_accuracy, best_param
                
def main():
    print(calc_all_perplexities("data_corrected/classification task"))
    

if __name__ == '__main__':
    main()
            
    