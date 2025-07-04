## 🧠 Emotion Classification from Text Using Naive Bayes and NLTK
This project uses machine learning and natural language processing to classify emotions in textual input. It leverages NLTK's pre-labeled Twitter dataset to train a Naive Bayes Classifier that can predict whether a given sentence expresses joy or sadness.

## 🎯 Key Features
✅ ML-based emotion prediction trained on real Twitter data
🔍 Tokenization and stopword filtering for clean input
🧠 Naive Bayes classifier for probabilistic prediction
📝 Interactive mode: enter your own sentence or auto-generate one
🎭 Returns emotion labels like "joy 😊" or "sadness 😢"

## 🔍 How It Works
Loads positive_tweets.json and negative_tweets.json from NLTK's twitter_samples.
Tokenizes and cleans each tweet using word_tokenize and stopwords.
Builds a feature set by mapping each word to a boolean presence flag.
Trains a Naive Bayes model with both positive and negative examples.
Accepts new text input from the user, cleans it, extracts features, and predicts the emotion.

## Results
![manu 5](https://github.com/user-attachments/assets/396b6112-3fea-4494-b706-db38bd4e9b1b)



## 📦 Installation
Install the necessary package:

bash
pip install nltk
Run the following once in your Python environment to download required corpora:

python
import nltk
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

## ▶️ Running the Project
Run the script:
bash : python script.py

Choose an input mode:
1 → Enter your own custom sentence
2 → Use a cheerful sample sentence

The model returns a predicted emotion based on trained probabilities.
