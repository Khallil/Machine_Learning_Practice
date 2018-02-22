from nltk.tokenize import word_tokenize

# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

# split into sentences
sentences = sent_tokenize(text)

# split into words
tokens = word_tokenize(text)

# enleve les mots qui contiennent des caractères out of A-Z
words = [word for word in tokens if word.isalpha()]

from nltk.corpus import stopwords
# enleve les "a","is","to"
# Attention utilise uniquement pour classer des documents
# Difficile de comprendre le sens sans ces mots
stop_words = stopwords.words( ' english ' )

from nltk.stem.porter import PorterStemmer
# Raccourci les mots à leurs sources example : fishing -> fish
# Tester si ça marche aussi en français
stemmed = [porter.stem(word) for word in tokens]





