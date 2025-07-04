import nltk
import random
from nltk.corpus import twitter_samples, stopwords
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier
from nltk.classify import apply_features
import string

# NLTK Resources
#nltk.download('twitter_samples')
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('punkt_tab')

#  Preprocessing 
def clean_tokens(tokens):
    stop_words = set(stopwords.words('english'))
    cleaned = []
    for word in tokens:
        word = word.lower()
        if word in stop_words or word in string.punctuation:
            continue
        cleaned.append(word)
    return cleaned

# Load tweets and label them
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

positive_tokens = [clean_tokens(word_tokenize(tweet)) for tweet in positive_tweets]
negative_tokens = [clean_tokens(word_tokenize(tweet)) for tweet in negative_tweets]

def build_dataset(tokens_list, label):
    return [({word: True for word in tokens}, label) for tokens in tokens_list]

positive_dataset = build_dataset(positive_tokens, 'joy 😊')
negative_dataset = build_dataset(negative_tokens, 'sadness 😢')

# combine and shuffle
all_data = positive_dataset + negative_dataset
random.shuffle(all_data)


train_set = apply_features(lambda d: d, all_data)
classifier = NaiveBayesClassifier.train(train_set)


mode = input(" Type '1' to enter your own text, or '2' for a sample: ")
if mode == '1':
    user_text = input("Enter your text: ")
else:
    user_text = "I just feel so happy and excited about the future!"

tokens = clean_tokens(word_tokenize(user_text))
features = {word: True for word in tokens}

# Prediction
emotion = classifier.classify(features)
print(f"\n🔍 Predicted Emotion: {emotion}")
