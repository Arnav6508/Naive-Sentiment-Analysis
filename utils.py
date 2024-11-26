import numpy as np
from preprocess import preprocess

def build_freqs(tweets, ys):
    pos_freqs = {}
    neg_freqs = {}
    
    for tweet,y in zip(tweets,ys):
        for word in preprocess(tweet):
            if(int(y) == 1): pos_freqs[word] = pos_freqs.get(word,0)+1
            else: neg_freqs[word] = neg_freqs.get(word,0)+1

    return pos_freqs, neg_freqs

def predict_tweet(tweet, freqs):
    tweet = preprocess(tweet)

    pos_freqs = freqs['pos_freqs']
    neg_freqs = freqs['neg_freqs']

    pos_unique = len(pos_freqs)
    neg_unique = len(neg_freqs)

    unique = set()
    for word in pos_freqs: unique.add(word)
    for word in neg_freqs: unique.add(word)
    tot_unique = len(unique)

    prob = 0
    for word in tweet:
        num = (pos_freqs.get(word,0)+1)/(pos_unique + tot_unique)
        denom = (neg_freqs.get(word,0)+1)/(neg_unique + tot_unique)
        prob += np.log(num/denom)
    
    if prob>=0: return 1
    else: return 0

def test(x, y, freqs):
    y_hat = []
    for tweet in x:
        pred = predict_tweet(tweet, freqs)
        y_hat.append(pred)

    accuracy = ((np.array(y_hat) == np.squeeze(y, axis = -1)).sum())/len(y_hat)
    return accuracy