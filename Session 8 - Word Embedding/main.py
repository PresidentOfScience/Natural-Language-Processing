# Embedding
# Vektorisasi -> mengubah kata-kata menjadi bentuk angka

# TF-IDF
# Word2Vec
# GloVe
# One Hot Encoding


# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "This is the first document",
    "This document is the second document",
    "And then, this is the third document.",
    "Last, this is the fourth document."
]

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print(tfidf_matrix.toarray())

# Text similarity menggunakan Cosine Similarity

feature_names = tfidf_vectorizer.get_feature_names_out()
print(feature_names)


# Word2Vec

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

tokenized = [word_tokenize(sent.lower()) for sent in documents]

model = Word2Vec(sentences=tokenized)
word_embedding = model.wv

print(word_embedding['document'])