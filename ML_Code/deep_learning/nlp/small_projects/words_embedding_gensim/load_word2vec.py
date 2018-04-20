from gensim.models import KeyedVectors
import pickle

# load the google word2vec model
filename = '/home/doudou/Documents/IA_others/we_models/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman','king'], negative=['man'], topn=1)

print(result)