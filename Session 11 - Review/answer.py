# Utilities Libraries
import numpy as np
import os
import pandas as pd
import string
import random
import csv
import pickle
import re

# Natural Language Processing Libraries
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Word Embedding Library
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define Constant Variable
ENGLISH_STOPWORDS = stopwords.words('english')
PUNCTUATIONS = string.punctuation
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()
FILE_PATH = "filtered_data.csv"
SEED_VALUE = 1234
NLP_ENG_MODEL = spacy.load('en_core_web_sm')

dataset = []
review_text = ""
classifier = 0
list_words = []
labeled = []

def init():
    global list_words
    global labeled
    
    random.seed(SEED_VALUE)
    
    data = []
    with open(FILE_PATH, encoding='UTF-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    
    random.shuffle(data)

    for d in data:
        sentences = d[1].lower()
        # Remove numbers from the text
        sentences = ''.join([i for i in sentences if not i.isdigit()])
        sentences = re.sub(r'\W+\d+', '', sentences)
        words = word_tokenize(sentences)
        words = [word for word in words if word not in ENGLISH_STOPWORDS]
        words = [STEMMER.stem(word) for word in words]
        words = [LEMMATIZER.lemmatize(word) for word in words]
        words = [word for word in words if word not in string.punctuation]
        words = [word for word in words if word.isalpha()]
        
        for w in words:
            list_words.append(w)
        labeled.append((d[0], d[1], d[2]))

    fd = FreqDist(list_words)
    list_words = [word for word, count in fd.most_common(100)]
    
def train_model():
    global dataset
    global labeled
    global classifier
    
    for title, sentence, label in labeled:
        words = word_tokenize(sentence)
        words = [word for word in words if word not in ENGLISH_STOPWORDS]
        words = [STEMMER.stem(word) for word in words]
        words = [LEMMATIZER.lemmatize(word) for word in words]
        words = [word for word in words if word not in string.punctuation]
        words = [word for word in words if word.isalpha()]
        
        try:
            if label == "NEGATIVE":
                newlabel = "negative"
            elif label == "POSITIVE":
                newlabel = "positive"
            else:
                print(f"  [!] WARNING: UNEXPECTED LABEL VALUE '{label}'")
                continue

            dict = {}
            for feature in list_words:
                key = feature
                value = feature in words
                dict[key] = value
            dataset.append((dict, newlabel))
        except ValueError:
            print(f"  [!] WARNING: COULD NOT CONVERT LABEL '{label}' TO AN CATEGORICAL VALUES.")
    
    percentage = int(len(dataset) * 0.85)

    training_data = dataset[:percentage]
    testing_data = dataset[percentage:]

    # Using Naive Bayes Classifier for model training
    classifier = NaiveBayesClassifier.train(training_data)

    print("  ", end = "")
    print("  MODEL TRAINING ACCURACY : " + str(accuracy(classifier, testing_data)*100) + " %")
    file = open("model.pickle", "wb")
    pickle.dump(classifier, file)
    file.close()
    
def writeReview():
    os.system("cls")
    global review_text

    while(True):
        review = input("  [#] INPUT YOUR REVIEW [MORE THAN 20 WORDS] : ")
        cleaned = [word for word in word_tokenize(review) if word not in string.punctuation]
        if len(cleaned) < 20:
            print("  [!] YOUR REVIEW MUST CONSIST OF AT LEAST 20 WORDS!")
        else:
            input("  [>] PRESS ENTER TO CONTINUE...")
            break
        
    review_text = review
    menu()
    
def analyzeReview():
    if(review_text == ''):
        print("  [!] YOU NEED TO WRITE A REVIEW FIRST!")
        input("  [>] PRESS ENTER TO CONTINUE...")
        menu()
    else:
        review = review_text.lower()
        words = word_tokenize(review)
        words = [word for word in words if word not in ENGLISH_STOPWORDS]
        words = [LEMMATIZER.lemmatize(word) for word in words]
        words = [word for word in words if word not in string.punctuation]
        words = [word for word in words if word.isalpha()]
        
        dict = {}
        for feature in list_words:
            key = feature
            value = feature in words
            dict[key] = value
        classification = classifier.classify(dict)
        return classification.upper()
        
def viewRecommendedMovie():
    os.system('cls')
    
    if(review_text == ''):
        print("  [!] YOU NEED TO WRITE A REVIEW FIRST!")
        input("  [>] PRESS ENTER TO CONTINUE...")
        menu()
    else:
        # Word Embedding (TF-IDF)
        tfidf_vectorizer = TfidfVectorizer(vocabulary=list_words)
        movie_tfidf_matrix = tfidf_vectorizer.fit_transform([review[1] for review in labeled])
        user_tfidf_vector = tfidf_vectorizer.transform([review_text])

        # Count cosine similarity
        cosine_similarities = cosine_similarity(user_tfidf_vector, movie_tfidf_matrix)
        
        top_indices = cosine_similarities.argsort(axis=1)[0][-2:][::-1]
        top_movies = [(labeled[i][0]) for i in top_indices] 
        
        # Return the result
        print('  TOP 2 MOVIE RECOMMENDATION FOR YOU:')
        for i in range(2):
            print(f'  {i+1}: {top_movies[i]}')
        
        print()    
        input("  [>] PRESS ENTER TO CONTINUE...")
        menu()

def viewNamedEntityRecognition():
    os.system('cls')
    
    global labeled
    
    # Extracting the review texts from the labeled data
    movie_review_text = [data[1] for data in labeled] 
    
    # Combine all review texts into one string
    # combined_movie_review_text = ' '.join(movie_review_text)
    
    # Combine all review texts into one string and remove symbols followed by numbers
    combined_movie_review_text = ' '.join([re.sub(r'\W+\d+', '', review) for review in movie_review_text])  

    doc = NLP_ENG_MODEL(combined_movie_review_text)
    categories = {}

    for ent in doc.ents:
        label = ent.label_
        if label not in categories:
            # Using a set to store unique entities
            categories[label] = set()  
        categories[label].add(ent.text)

    print("  CATEGORIZED NAMED ENTITIES:")
    for label, entities in categories.items():
        print(f"  {label}: {', '.join(entities)}")

    input("  [>] PRESS ENTER TO CONTINUE...")
    menu()
        
def menu():
    os.system("cls")
    print("  MOVIE RECOMMENDATION APPLICATION BASED ON REVIEWS")
    print("  YOUR REVIEW : ", "NO REVIEW" if not review_text else review_text)
    print("  YOUR REVIEW CATEGORY : ", "UNKNOWN" if not review_text else analyzeReview())
    print("  1. WRITE YOUR REVIEW")
    print("  2. VIEW MOVIE RECOMMENDATION")
    print("  3. VIEW NAMED ENTITIES RECOGNITION")
    print("  4. EXIT")
    
    while True:
        try:
            opt = int(input("  >> "))
            if opt < 1 or opt > 4:
                print("  [!] PLEASE ENTER A NUMBER BETWEEN 1 AND 4 (INCLUSIVE)!")
            else:
                # Break out of the loop if the input is valid
                break  
        except ValueError:
            print("  [!] INVALID INPUT. PLEASE ENTER A NUMBER!")
            
    if int(opt) == 1:
        writeReview()
    elif int(opt) == 2:
        viewRecommendedMovie()
    elif int(opt) == 3:
        viewNamedEntityRecognition()
    elif int(opt) == 4:
        os.system("cls")
        print("  [>] THANK YOU FOR USING THIS APPLICATION!")
        return
    
def main():
    init()
    os.system("cls")
    
    global classifier
    
    if os.path.isfile("model.pickle"):
        file = open("model.pickle", "rb")
        classifier = pickle.load(file)
        file.close()
        
        print("  [>] LOAD MODEL COMPLETE...")
    else:
        print("  [>] TRAINING...")
        train_model()
        print("  [>] TRAINING MODEL COMPLETE...")
    input("  [>] PRESS ENTER TO CONTINUE...")
    menu()
    
main()