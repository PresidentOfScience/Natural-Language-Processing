import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
import random
from nltk.classify import NaiveBayesClassifier, accuracy
import pickle

def remove_symbols(word_list):
    punctuations = string.punctuation
    removed_symbols = []
    for word in word_list:
        word = "".join([c for c in word if c not in punctuations])
        removed_symbols.append(word)
    
    return removed_symbols

def preprocess_data(text):
    text = text.lower()

    # Tokenize
    word_list = word_tokenize(text)

    # Remove Stopwords
    eng_stopwords = stopwords.words('english')
    word_list = [word for word in word_list if word not in eng_stopwords]

    # Remove Numeric Words
    word_list = [word for word in word_list if word.isalpha()]

    # Remove Symbols
    word_list = remove_symbols(word_list)

    # Stemming
    stemmer = SnowballStemmer('english')
    word_list = [stemmer.stem(word) for word in word_list]

    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    word_list = [lemmatizer.lemmatize(word) for word in word_list]

    return word_list

def extract_features(document):
    all_words = []
    for text in document:
        clean_word_list = preprocess_data(text)
        all_words.extend(clean_word_list)
    
    fd = FreqDist(all_words)
    common_words = [word for word, count in fd.most_common(500)]
    common_words = list(set(common_words))

    return common_words

def extract_dataset():
    PATH = 'dataset.csv'
    dataset = pd.read_csv(PATH)

    # Untuk membuat dictionary
    word_dictionary = extract_features(dataset['Review'])

    document = []
    for index, data in dataset.iterrows():
        features = {}
        review = preprocess_data(data['Review'])
        for feature in word_dictionary:
            key = feature
            value = feature in review
            features[key] = value
        
        label = 'positive' if data['Rating'] >= 3 else 'negative'

        document.append((features, label))
    
    return document

def train_data(document):
    random.shuffle(document)
    training_amount = int(len(document) * 0.7)
    training_data = document[:training_amount]
    testing_data = document[training_amount:]

    classifier = NaiveBayesClassifier.train(training_data)
    classifier.show_most_informative_features(5)

    print(f"Accuracy: {accuracy(classifier, testing_data) * 100}%")

    # Save model
    file = open('model.pickle', 'wb')
    pickle.dump(classifier, file)
    file.close()

def load_model():
    try:
        with open('model.pickle', 'rb') as file:
            classifier = pickle.load(file)
        print("Model Load Complete...")
        input("Press enter to continue...")
    except:
        dataset = extract_dataset()
        classifier = train_data(dataset)
        print("Training Model Complete...")
        input("Press enter to continue...")
    
    return classifier

def input_analyze_review(classifier):
    review = input("Write your review: ")
    review = preprocess_data(review)

    dataset_file_path = 'dataset.csv'
    dataset = pd.read_csv(dataset_file_path)

    important_words = extract_features(dataset['Review'])
    features = {}
    for feature in important_words:
        key = feature
        value = feature in review
        features[key] = value
    
    return classifier.classify(features)

if __name__ == "__main__":
    input_analyze_review(load_model())