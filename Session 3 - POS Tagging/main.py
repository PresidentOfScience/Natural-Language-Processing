import nltk
nltk.download('averaged_perceptron_tagger')

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

sentence = "I eat a fried rice for breakfast"

word_list = word_tokenize(sentence)
pos_tagged = pos_tag(word_list)

print(pos_tagged)

# Frequency Distribution -> menghitung distribusi frekuensi sebuah kata dalam sebuah teks

from nltk.probability import FreqDist

sentence = "I am a human because I have two legs and two hands"

word_list = word_tokenize(sentence)
fd = FreqDist(word_list)

print(fd.most_common())
print(fd.most_common(3))