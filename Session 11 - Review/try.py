import os
import re
import csv
import pickle
import random
import string
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FILE_PATH = 'filtered_data.csv'
SEED_VALUE = 42
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()
ENG_STOPWORDS = stopwords.words('english')
PUNCTUATIONS = string.punctuation
NLP_ENG_MODEL = spacy.load('en_core_web_sm')

dataset = []
word_list = []
labeled = []
review_text = ""
classifier = 0

def init():
    global word_list, labeled

    random.seed(SEED_VALUE)

    data = []
    with open(FILE_PATH, encoding='UTF-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    random.shuffle(data)

    for d in data:
        sentence = d[1].lower()
        sentence = ''.join([i for i in sentence if not i.isdigit()])
        sentence = re.sub(r'\W+\d+', '', sentence)
        words = word_tokenize(sentence)
        words = [word for word in words if word not in ENG_STOPWORDS]
        words = [word for word in words if word not in PUNCTUATIONS]
        words = [word for word in words if word.isalpha()]
        words = [STEMMER.stem(word) for word in words]
        words = [LEMMATIZER.lemmatize(word) for word in words]

        for w in words:
            word_list.append(w)
        labeled.append((d[0], d[1], d[2]))

    fd = FreqDist(word_list)
    word_list = [word for word, _ in fd.most_common(10000)]

def train_model():
    global dataset, classifier, labeled

    for title, sentence, label in labeled:
        words = word_tokenize(sentence)
        words = [word for word in words if word not in ENG_STOPWORDS]
        words = [word for word in words if word not in PUNCTUATIONS]
        words = [word for word in words if word.isalpha()]
        words = [STEMMER.stem(word) for word in words]
        words = [LEMMATIZER.lemmatize(word) for word in words]

        try:
            if label == 'POSITIVE':
                newlabel = 'positive'
            elif label == 'NEGATIVE':
                newlabel = 'negative'
            else:
                print(f'There is no existing label {label}')
                continue

            dict = {}
            for feature in word_list:
                key = feature
                value = feature in words
                dict[key] = value
            dataset.append((dict, newlabel))
        except ValueError:
            print(f'Cannot convert label {label} into categorical value')
        
    percentage = int(len(dataset) * 0.75)
    training_data = dataset[:percentage]
    testing_data = dataset[percentage:]

    classifier = NaiveBayesClassifier.train(training_data)
    print(f'MODEL ACCURACY = {accuracy(classifier, testing_data)*100}%')
    file = open('models.pickle', 'wb')
    pickle.dump(classifier, file)
    file.close()

def write_review():
    os.system('cls')
    global review_text

    review = input('[>] Input text (minimal 20 words): ')
    cleaned = [word for word in word_tokenize(review) if word not in string.punctuation]
    while(True):
        if len(cleaned) < 20:
            print('THERE MUST BE 20 WORDS REVIEW!')
        else:
            input('Press enter to continue')
            break
    
    review_text = review
    menu()

def analyze_review():
    if review_text == '':
        print('YOU NEED TO WRITE REVIEW FIRST!')
        input('PRESS ENTER TO CONTINUE...')
        menu()
    else:
        review = review_text.lower()
        words = word_tokenize(review)
        words = [word for word in words if word not in ENG_STOPWORDS]
        words = [word for word in words if word not in PUNCTUATIONS]
        words = [word for word in words if word.isalpha()]
        words = [STEMMER.stem(word) for word in words]
        words = [LEMMATIZER.lemmatize(word) for word in words]

        dict = {}
        for feature in word_list:
            key = feature
            value = feature in words
            dict[key] = value
        classification = classifier.classify(dict)
        return classification.upper()

def view_recommendation_movies():
    os.system('cls')

    if review_text == '':
        print('YOU NEED TO WRITE REVIEW FIRST!')
        input('PRESS ENTER TO CONTINUE...')
        menu()
    else:
        tfidf_vectorizer = TfidfVectorizer(vocabulary=word_list)
        movie_matrix = tfidf_vectorizer.fit_transform([review[1] for review in labeled])
        user_vector = tfidf_vectorizer.transform([review_text])

        cosine_similarities = cosine_similarity(user_vector, movie_matrix)

        top_indices = cosine_similarities.argsort(axis=1)[0][-2:][::-1]
        top_movies = [(labeled[i][0]) for i in top_indices]

        print('TOP 2 MOVIES RECOMMENDED FOR YOU!')
        for i in range(2):
            print(f'{i+1}. {top_movies[i]}')
        input('PRESS ENTER TO CONTINUE...')
        menu()

def view_NER():
    os.system('cls')
    global labeled

    movie_review_text = [data[1] for data in labeled]
    combined_movie_review_text = ' '.join([re.sub(r'\W+\d+', '', review) for review in movie_review_text])

    categories = {}
    doc = NLP_ENG_MODEL(combined_movie_review_text)

    for ent in doc.ents:
        label = ent.label_
        if label not in categories:
            categories[label] = set()
        categories[label].add(ent.text)
    
    print('CATEGORIZED NAMED ENTITIES')
    for label, entities in categories.items():
        print(f"{label}: {', '.join(entities)}")
    
    input('PRESS ENTER TO CONTINUE...')
    menu()

def menu():
    os.system('cls')
    print('MOVIE RECOMMENDATION BASED ON REVIEWS')
    print('YOUR REVIEW: ', 'NO REVIEW' if not review_text else review_text)
    print('YOUR REVIEW CATEGORY: ', 'UNKNOWN' if not review_text else analyze_review())
    print('1. WRITE YOUR REVIEW')
    print('2. VIEW MOVIE RECOMMENDATION')
    print('3. VIEW NAMED ENTITY RECOGNITION')
    print('4. EXIT')
    
    while True:
        opt = int(input('> '))
        try:
            if opt < 1 or opt > 4:
                print('INPUT MUST BE BETWEEN 1 TO 4 (INCLUSIVE)')
            else:
                break
        except ValueError:
            print('INVALID INPUT! TRY TO INPUT NUMBER ONLY!')
    
    if opt == 1:
        write_review()
    elif opt == 2:
        view_recommendation_movies()
    elif opt == 3:
        view_NER()
    else:
        print('Thank you for using this application!')
        return

def main():
    init()
    os.system('cls')

    global classifier

    if os.path.isfile('models.pickle'):
        file = open('models.pickle', 'rb')
        classifier = pickle.load(file)
        file.close()
        print('LOAD MODEL COMPLETED.')
    else:
        print('TRAINING...')
        train_model()
        print('TRAIN MODEL COMPLETED.')
    
    input('PRESS ENTER TO CONTINUE...')
    menu()

main()