import string

# sentence = ""

# while len(sentence) < 10:
#     sentence = input("Please input a sentence: ")

# while 'In my opinion,' not in sentence:
#     sentence = input("Please input a sentence: ")

sentence = ""
count = 0

# while count < 3:
#     sentence = input("Please input a sentence: ")
#     sentence = sentence.strip()
#     word_list = sentence.split(' ')
#     count = len(word_list)

# numeric = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
while True:
    sentence = input("Please input a sentence: ")
    alpha = False
    numeric = False
    for character in sentence:
        if character.isalpha():
            alpha = True
        elif character.isnumeric():
            numeric = False
        else:
            alpha = False
            numeric = False
            break
    if alpha == True and numeric == True:
        break