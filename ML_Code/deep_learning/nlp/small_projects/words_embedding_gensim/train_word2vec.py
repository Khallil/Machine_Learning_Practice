from gensim.models import Word2Vec
import gensim.models.word2vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import time

# Training Word2Vec
# Visualize in 2D the embedding

assert gensim.models.word2vec.FAST_VERSION

start_time = time.time()

# define training data
sentences = [['yet', 'another', 'sentence'],
			['this', 'is', 'the', 'second', 'sentence'],
			['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['one', 'more', 'sentence'],			
			['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1,workers=4)

print("--- %s seconds ---" % (time.time() - start_time))


# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])

# save model # en 2 formats differents
model.wv.save_word2vec_format('model.txt',binary=False)
model.save('model.bin')

# Visualiser l'embedding
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
