import nltk
import numpy as np

nltk.download('twitter_samples')

from nltk.corpus import twitter_samples 

training_size = 4000
testing_size = 1000

def load_data():

    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    test_pos = all_positive_tweets[training_size:]
    train_pos = all_positive_tweets[:training_size]
    test_neg = all_negative_tweets[training_size:]
    train_neg = all_negative_tweets[:training_size]

    train_x = train_pos + train_neg 
    test_x = test_pos + test_neg

    train_y = np.append(np.ones((training_size,1)), np.zeros((training_size,1)), axis = 0)
    test_y = np.append(np.ones((testing_size,1)), np.zeros((testing_size,1)), axis = 0)

    return train_x, test_x, train_y, test_y
