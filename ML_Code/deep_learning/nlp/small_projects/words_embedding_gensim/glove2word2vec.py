from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = '/home/doudou/Documents/IA_others/we_models/glove/glove.6B.100d.txt'
word2vec_output_file = '/home/doudou/Documents/IA_others/we_models/glove/glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)