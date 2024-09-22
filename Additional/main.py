import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
import string
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.probability import FreqDist
import random
from nltk.classify import NaiveBayesClassifier, accuracy
import pickle

eng_stopwords = stopwords.words('english')
punctuation_list = string.punctuation
stemming = SnowballStemmer('english')
wnl = WordNetLemmatizer()

def remove_stopwords(word_list):
    return [word for word in word_list if word not in eng_stopwords]

def remove_punctuation(word_list):
    return [word for word in word_list if word not in punctuation_list]

def remove_number(word_list):
    return [word for word in word_list if word.isalpha()]

def stemming_words(word_list):
    return [stemming.stem(word) for word in word_list]

def get_tag(tag):
    if tag == 'jj':
        return 'a'
    elif tag in ['vb', 'nn', 'rb']:
        return tag[0]
    else:
        return None

def lemmatizing_words(word_list):
    lemmatizing = []
    tagging = pos_tag(word_list)
    for word, tag in tagging:
        label = get_tag(tag.lower())
        if label != None:
            lemmatizing.append(wnl.lemmatize(word, label))
        else:
            lemmatizing.append(wnl.lemmatize(word))
    return lemmatizing

def training_model():
    dataset = pd.read_csv('./IMDB Dataset.csv').sample(n=200)
    review_list = dataset['review'].to_list()
    label_list = dataset['sentiment'].to_list()

    word_list = []

    for sentence in review_list:
        words = word_tokenize(sentence)
        for word in words:
            word_list.append(word.lower())
    
    word_list = remove_stopwords(word_list)
    word_list = remove_punctuation(word_list)
    word_list = remove_number(word_list)
    word_list = stemming_words(word_list)
    word_list = lemmatizing_words(word_list)

    fd = FreqDist(word_list)
    word_features = [word for word, count in fd.most_common(n=100)]

    labeled_data = list(zip(review_list, label_list))

    feature_sets = []

    for sentence, label in labeled_data:
        features = {}
        check_list = word_tokenize(sentence)
        check_list = remove_stopwords(check_list)
        check_list = remove_punctuation(check_list)
        check_list = remove_number(check_list)
        check_list = stemming_words(check_list)
        check_list = lemmatizing_words(check_list)

        for word in word_features:
            features[word] = (word in check_list)
        
        feature_sets.append((features, label))
    
    random.shuffle(feature_sets)
    train_count = int(len(feature_sets)*0.8)
    train_dataset = feature_sets[0:train_count]
    test_dataset = feature_sets[train_count:]

    classifier = NaiveBayesClassifier.train(train_dataset)
    classifier.show_most_informative_features(n=5)
    print(f"Training accuracy: {accuracy(classifier, test_dataset)*100} %")

    file = open('model.pickle', 'wb')
    pickle.dump(classifier, file)
    file.close()

    return classifier

try:
    file = open('model.pickle', 'rb')
    classifier = pickle.load(file)
    file.close()
except:
    print("No model")
    classifier = training_model()
    print("Training Model Completed...")
    input("Press enter to continue...")

reviews = ""

def show_menu():
    print("Menu:")
    print("1. Write reviews")
    print("2. Analyze reviews")
    print("3. Exit")

def menu_1():
    global reviews
    if reviews != "":
        print("You already input the reviews!")
        return
    
    while True:
        input_reviews = input("Input reviews: ")
        length = len(input_reviews.split(' '))
        if length < 5:
            print("The reviews must contain at least 5 words.")
        else:
            reviews = input_reviews
            break
    print("Successful saved the reviews")

def get_pos_tag():
    r_word_list = word_tokenize(reviews)
    tagging_list = pos_tag(r_word_list)
    print("Reviews part of speech tag")
    for idx, (word, tag) in enumerate(tagging_list):
        print(f"{idx+1}. {word} : {tag}")
    input("Press enter to continue...")

def get_synonym_antonym():
    r_word_list = word_tokenize(reviews)
    for word in r_word_list:
        print(f"Word: {word}")
        print("-----------------------------------------")
        synonym_list = []
        antonym_list = []
        synsets = wordnet.synsets(word)
        for syn in synsets:
            for lemma in syn.lemmas():
                synonym_list.append(lemma.name())
                for antonym in lemma.antonyms():
                    antonym_list.append(antonym.name())
        
        if len(synonym_list) != 0:
            print("Synonym")
            print(f"(+) {synonym_list[0]}")
        else:
            print("The word doesn't have any synonym")

        if len(antonym_list) != 0:
            print("      Antonym")
            print(f"     (-) {antonym_list[0]}")
        else:
            print("The word doesn't have any antonym")
    print("\n")

def get_label():
    print(f"Reviews Category: {classifier.classify(FreqDist(word_tokenize(reviews)))}")
    input("Press enter to continue...")

def menu_2():
    global reviews
    if reviews == "":
        print("Please input the reviews first")
        return
    else:
        get_pos_tag()
        get_synonym_antonym()
        get_label()
        reviews = ""

while True:
    show_menu()
    choice = input("Input choice: ")
    if choice == '1':
        menu_1()
    elif choice == '2':
        menu_2()
    elif choice == '3':
        print("Thank you")
        break
    else:
        print("Invalid input")