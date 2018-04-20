# --coding:utf-8--#

from gensim.models import KeyedVectors
from googletrans import Translator
import time

translator = Translator()

def print_result(result,x_word):
    print x_word
    for r in result:
        print " ",r

# convert glove to word2vec format

# load the converted model
filename = '/home/doudou/Documents/IA_others/we_models/glove/glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

# bon + (pollution - ecologie) = mauvais
start_time = time.time()

print("Pollution result : ")
print translator.translate("J'aime l'écologie", dest='en').text


# result = mauvais -> donc pollution est mauvais pour ecologie
# bon + (pollution - ecologie) = mauvais
x_word = "king"
result = model.most_similar(positive=['woman','king'], negative=['man'], topn=3)
print_result(result,x_word)
# result = mauvais -> donc pollution est mauvais pour ecologie
x_word = "negative"
result = model.most_similar(positive=['ecology',x_word], negative=['pollution'], topn=3)
print_result(result,x_word)

x_word = "positive"
result = model.most_similar(positive=['ecology',x_word], negative=['pollution'], topn=3)
print_result(result,x_word)

# si un des deux possède bad ou good dans leur liste alors on assigne positif ou negatif
# a celui trouvé comme étant bad ou good
'''print("Oxygen result : ")
x_word = "good"
result = model.most_similar(positive=['oxygen',x_word], negative=['ecology'], topn=3)
print_result(result,x_word)
# result = mauvais -> donc pollution est mauvais pour ecologie
# bon + (pollution - ecologie) = mauvais
x_word = "bad"
result = model.most_similar(positive=['oxygen',x_word], negative=['ecology'], topn=3)
print_result(result,x_word)
print("--- %s seconds ---" % (time.time() - start_time))
'''