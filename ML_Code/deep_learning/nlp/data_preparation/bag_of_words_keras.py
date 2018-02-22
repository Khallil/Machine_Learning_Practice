from keras.preprocessing.text import text_to_word_sequence

# define the document
text = 'The quick brown fox jumped over the lazy dog.'

# tokenize the document
result = text_to_word_sequence(text)
#tex to word sequence step by step
#Splits words by space
#Filters out punctuation
#Converts text to lowercase (lower=True)
print(result)

from keras.preprocessing.text import one_hot
result = one_hot(text, round(vocab_size*1.3))
[5, 9, 8, 7, 9, 1, 5, 3, 8]

from keras.preprocessing.text import hashing_trick
result = hashing_trick(text, round(vocab_size*1.3), hash_function= ' md5 ' )
[6, 4, 1, 2, 7, 5, 6, 2, 6]