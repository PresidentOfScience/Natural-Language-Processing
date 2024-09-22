import string
import re

sentence = "I am a human because I have two legs and two hands."
sentence = re.sub('[^A-Za-z ]', '', sentence)
print(sentence)

word_list = sentence.split(" ")

# for p in string.punctuation:
#     for word in word_list:
#         if p in word:
#             word.replace(p, '')

print(word_list)

word_dict = {}

for word in word_list:
    if word in word_dict.keys():
        word_dict[word] += 1
    else:
        word_dict[word] = 1

print(word_dict)

word_dict = dict(sorted(word_dict.items(), key=lambda x:x[1], reverse=True))
print(word_dict)