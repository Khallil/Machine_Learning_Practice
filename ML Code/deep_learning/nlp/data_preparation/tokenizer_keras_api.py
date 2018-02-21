from keras.preprocessing.text import Tokenizer

# Bag of Words
# define 5 documents
docs = [ 'Well done!' ,
'Good work' ,
'Great effort' ,
'nice work' ,
'Excellent!']

# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)

encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)

# Voir le n-grams (groups de mots) pour réduire la taille des vecteurs
# Plus efficace pour document classification