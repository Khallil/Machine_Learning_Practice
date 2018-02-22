from sklearn.feature_extraction.text import CountVectorizer

# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())

from sklearn.feature_extraction.text import TfidfVectorizer
[ 1.69314718 1.28768207 1.28768207 1.69314718 1.69314718 1.69314718
1.69314718 1. ]

from sklearn.feature_extraction.text import HashingVectorizer
[[ 0.0.0.0.0 0.33333333.0.-0.33333333 0.33333333 0.0.0.0.0.-0.33333333 0.0.-0.66666667 0.]]
# for a 20 element spart array

