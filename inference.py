import pickle
from utils import predict_tweet

def predict_one_tweet(tweet):
    with open('freqs.pkl', 'rb') as file:
        freqs = pickle.load(file)

    pred = predict_tweet(tweet, freqs)

    if pred == 1: print('Positive Sentiment')
    else: print('Negative Sentiment')
