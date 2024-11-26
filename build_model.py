import pickle
from load import load_data
from utils import build_freqs
from inference import test

def build_model():
    train_x, test_x, train_y, test_y = load_data()
    pos_freqs, neg_freqs = build_freqs(train_x, train_y)

    freqs = {"pos_freqs": pos_freqs,
             "neg_freqs": neg_freqs
            }

    with open("freqs.pkl", "wb") as file:
        pickle.dump(freqs, file)

    accuracy = test(test_x, test_y, freqs)
    print('Accuracy:', accuracy)

if __name__ == "__main__":
    build_model()
