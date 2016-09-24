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
                    
        summation += (-math.log(prob_of_token))
        
    pp = math.exp(1/token_length * summation)
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
            else:

                if prev_word in bigram_probs:
                    prob_of_pair = bigram_probs[prev_word]
                else:
                    prob_of_pair = bigram_probs["unk"]

            summation += (-math.log(prob_of_pair))
            
        prev_word = word
    pp = math.exp(1/token_length * summation)
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
            if triple in trigram_probs:
                prob_of_triple = trigram_probs[triple]
            else:
                prev_bigram = (two_prev_word, prev_word)
                if prev_bigram in trigram_probs:
                    prob_of_triple = trigram_probs[prev_bigram]
                else:
                    prob_of_triple = trigram_probs["unk"]
                
            summation += (-math.log(prob_of_triple))
        
        two_prev_word = prev_word        
        prev_word = word
        
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
         

def classify_topics_validation(topics_dir):
    best_accuracy_unigram = 0
    best_accuracy_bigram = 0
    best_accuracy_trigram = 0
    
    best_param_unigram = 0
    best_param_bigram = 0
    best_param_trigram = 0
    
    # Tune...
    for i in range(3,11):
        smoothing_param = i
        
        topic_to_ngram_dict = get_topic_to_ngram_dict(topics_dir, smoothing_param)
            
        # For computing accuracy
        num_correct_unigram = 0
        num_correct_bigram = 0
        num_correct_trigram = 0
        
        num_total = 0
        
        for topic in os.listdir(topics_dir):
            if topic == "test_for_classification":
                continue
            
            for validation_file in os.listdir(
                os.path.join(topics_dir, topic, "validation_docs")):
                
                validation_file = os.path.join(topics_dir, topic, "validation_docs", validation_file)
                
                (best_unigram_topic_guess, best_bigram_topic_guess, best_trigram_topic_guess) = predict_topic(topics_dir, validation_file, topic_to_ngram_dict, smoothing_param)

                if topic == best_unigram_topic_guess:
                    num_correct_unigram += 1
                    
                if topic == best_bigram_topic_guess:
                    num_correct_bigram += 1
                    
                if topic == best_trigram_topic_guess:
                    num_correct_trigram += 1
                    
                num_total += 1
                
        accuracy_unigram = num_correct_unigram / num_total
        accuracy_bigram = num_correct_bigram / num_total
        accuracy_trigram = num_correct_trigram / num_total
        
        print("unigram " + str(i) + " accuracy is " + str(accuracy_unigram))
        print("bigram " + str(i) + " accuracy is " + str(accuracy_bigram))
        print("trigram " + str(i) + " accuracy is " + str(accuracy_trigram))
        
        if accuracy_unigram > best_accuracy_unigram:
            best_accuracy_unigram = accuracy_unigram
            best_param_unigram = smoothing_param
            
        if accuracy_bigram > best_accuracy_bigram:
            best_accuracy_bigram = accuracy_bigram
            best_param_bigram = smoothing_param
            
        if accuracy_trigram > best_accuracy_trigram:
            best_accuracy_trigram = accuracy_trigram
            best_param_trigram = smoothing_param
            
    return best_accuracy_unigram, best_param_unigram, best_accuracy_bigram, best_param_bigram, best_accuracy_trigram, best_param_trigram



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
            
            (_, best_bigram_topic_guess, _) = predict_topic(topics_dir, test_file, topic_to_ngram_dict, smoothing_param)
            csv_writer.writerow([file, topic_to_num[best_bigram_topic_guess]])


                
def main():


    # Uncomment this to calculate the perplexity for each corpus for each test file
    # This will print out a Python Dict with each file name and corpus mapping to a perplexity number
#     print(calc_all_perplexities("data_corrected/classification task"))

    count=0
    avg_dict = {}
    pp_dict = calc_all_perplexities("data_corrected/classification task")
    for _, topic_to_pp_dict in pp_dict.items():
        for topic, (unigram, bigram, trigram) in topic_to_pp_dict.items():
            
            if topic not in avg_dict:
                avg_dict[topic] = (0,0,0)

            curr_unigram = avg_dict[topic][0]
#             print("u " + str(unigram))
#             print("c " + str(curr_unigram))
#             print("sum " + str(curr_unigram + unigram))
#             
            
            curr_bigram = avg_dict[topic][1]
            curr_trigram = avg_dict[topic][2]
            
            avg_dict[topic] = (curr_unigram + unigram, curr_bigram + bigram, curr_trigram + trigram)
#             print(avg_dict[topic])
#             
#             if count == 50:
#                 return
             
        count += 1
            # Could add trigram too if we want
            
    for topic, (unigram, bigram, trigram) in avg_dict.items():
        avg_dict[topic] = (unigram / count, bigram / count, trigram / count)
    print(avg_dict)    

    # Uncomment and change filename to predict the topic of a certain file in the test_for_classification directory
#     filename = "file_186.txt"
#     print(predict_topic("data_corrected/classification task", 
#                         os.path.join("data_corrected/classification task/test_for_classification", filename)))

    # Uncomment to print out the accuracy and best smoothing parameters when classifying the validation set
#     print(classify_topics_validation("data_corrected/classification task"))

    # Uncomment to classify the test documents and this creates the CSV that is submitted to Kaggle
#     classify_topics_test("data_corrected/classification task", 10)

if __name__ == '__main__':
    main()
            
    