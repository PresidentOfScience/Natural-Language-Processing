import os
import pickle
import csv
import random
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier, accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_PATH = "filtered_data.csv"
SEED_VALUE = 42

word_dictionary = []
labeled_dataset = []

def read_csv():
    global word_dictionary, labeled_dataset
    dataset = []
    with open(DATASET_PATH, encoding='UTF-8') as file:
        reader = csv.reader(file)
        for row_data in reader:
            dataset.append(row_data)
    
    random.seed(SEED_VALUE)
    random.shuffle(dataset)

    for row in dataset:
        # Indeks 0 -> Movie Title
        # Indeks 1 -> Review Sentence
        # Indeks 2 -> Label
        sentence = row[1]
        # Clean dataset
        sentence = re.sub('[^A-Za-z ]', '', sentence)
        # Tokenize dataset
        word_list = word_tokenize(sentence)
        # Remove stopwords
        english_stopwords = stopwords.words('english')
        word_list = [word for word in word_list if word not in english_stopwords]
        # Stemming
        porter_stemmer = PorterStemmer()
        word_list = [porter_stemmer.stem(word) for word in word_list]
        # Lemmatizing
        lemmatizer = WordNetLemmatizer()
        word_list = [lemmatizer.lemmatize(word) for word in word_list]

        word_dictionary.extend(word_list)
        labeled_dataset.append((row[0], row[1], row[2]))

    fd = FreqDist(word_dictionary)
    word_dictionary = [word for word, count in fd.most_common(100)]

def classify():
    global labeled_dataset, word_dictionary
    dataset = []
    for title, sentence, label in labeled_dataset:
        # Clean dataset
        sentence = re.sub('[^A-Za-z ]', '', sentence)
        # Tokenize dataset
        word_list = word_tokenize(sentence)
        # Remove stopwords
        english_stopwords = stopwords.words('english')
        word_list = [word for word in word_list if word not in english_stopwords]
        # Stemming
        porter_stemmer = PorterStemmer()
        word_list = [porter_stemmer.stem(word) for word in word_list]
        # Lemmatizing
        lemmatizer = WordNetLemmatizer()
        word_list = [lemmatizer.lemmatize(word) for word in word_list]
        
        # Label
        if label == 'POSITIVE':
            new_label = 1
        elif label == 'NEGATIVE':
            new_label = 0
        
        dict = {}
        for feature in word_dictionary:
            key = feature
            value = feature in word_list
            dict[key] = value
        dataset.append((dict, new_label))

    # Split Training-Testing
    training_amount = int(len(dataset) * 0.75)
    training_data = dataset[:training_amount]
    testing_data = dataset[training_amount:]

    classifier = NaiveBayesClassifier.train(training_data)
    print(f'Model accuracy: {accuracy(classifier, testing_data) * 100}%')

    file = open('model.pickle', 'wb')
    pickle.dump(classifier, file)
    file.close()

def give_recommendation():
    global word_dictionary, labeled_dataset
    text_review = "I like action movie and comedy movie"

    vectorizer = TfidfVectorizer(vocabulary=word_dictionary)
    all_matrix = vectorizer.fit_transform([row[1] for row in labeled_dataset])
    user_review_matrix = vectorizer.transform([text_review])

    cosine_sim = cosine_similarity(all_matrix, user_review_matrix)
    top_indexes = cosine_sim.argsort(axis=1)[0][-2:][::-1]

    top_movies = [(labeled_dataset[i][0]) for i in top_indexes]
    print(" TOP 2 MOVIE RECOMMENDATION FOR YOU:")
    for i in range(2):
        print(f"{i+1}: {top_movies[i]}")

def check_model():
    if os.path.isfile("model.pickle"):
        file = open("model.pickle", "rb")
        classifier = pickle.load(file)
        file.close()
        print(" [>] LOAD MODEL COMPLETED!")
    else:
        print(" [>] TRAIN MODEL . . .")


if __name__ == "__main__":
    # check_model()
    read_csv()
    # print(word_dictionary)
    # classify()
    give_recommendation()