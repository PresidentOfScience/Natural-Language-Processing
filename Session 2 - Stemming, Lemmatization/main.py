# Stemming -> Menghilangkan imbuhan
# Contoh: memakan -> makan, writes -> write
# Berdasarkan pola, e/es/ing akan dihilangkan

# PorterStemmer, SnowballStemmer, LancasterStemmer

from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")
lancaster_stemmer = LancasterStemmer()

words = ['dances', 'writting', 'written', 'writes', 'wrote', 'program', 'programmer', 'programs', 'best', 'computing', 'computer']

for word in words:
    print(f"{word}")
    print(f"Porter: {porter_stemmer.stem(word)}")
    print(f"Snowball: {snowball_stemmer.stem(word)}")
    print(f"Lancaster: {lancaster_stemmer.stem(word)}")
    print("===================================")


# Lemmatization -> akan mengembalikan ke kata dasarnya, berbasis dictionary

import nltk
nltk.download("wordnet")

from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

words = ['caring', 'running', 'better', 'troubled', 'finally', 'running', 'was']

for word in words:
    print(f"{word}")
    print(f"Noun: {wnl.lemmatize(word, pos='n')}")    # n -> kata benda
    print(f"Verb: {wnl.lemmatize(word, pos='v')}")    # v -> kata kerja
    print(f"Adjective: {wnl.lemmatize(word, pos='a')}")    # a -> kata sifat
    print(f"Adverb: {wnl.lemmatize(word, pos='r')}")    # r -> kata keterangan
    print("===================================")

sentence = "I was eating drinking seeing saw at a restaurant and suddenly saw a woman whose face was familiar to me. Suddenly, she accidentally stumbled upon me."
punctuations = "?!.,:"

sentence_word = nltk.word_tokenize(sentence)
print(sentence_word)

# Remove Punctuations
for word in sentence_word:
    if word in punctuations:
        sentence_word.remove(word)

print(sentence_word)

# Remove Stopwords
from nltk.corpus import stopwords

list_stopwords = stopwords.words("english")
sentence_word = [word for word in sentence_word if word.lower() not in list_stopwords]

print(sentence_word)

for word in sentence_word:
    print("{0:50} {1:50}".format(word, wnl.lemmatize(word, pos='v')))