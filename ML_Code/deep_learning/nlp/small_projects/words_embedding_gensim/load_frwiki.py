# --coding: utf-8--#

from gensim.models import KeyedVectors

# Comment load un split save de gensim ?
# at +0.6 - > veut dire que le résultat est très certain

def print_result(result,x_word):
    print x_word
    for r in result:
        print " ",r


# "ok" is not in vocabulary
# 'nucléaire" not in vocabulary
#filename = '/home/doudou/Documents/IA_others/we_models/frwiki/frwiki.gensim'
#fr_model = KeyedVectors.load(filename)

# frWiki2Vec 600 millions words
'''# 'ecologie est pas dans le vocabulaire'
filename = '/home/doudou/Documents/IA_others/we_models/frwiki/frWiki_no_lem_no_postag_no_phrase_1000_cbow_cut100.bin'
fr_model = KeyedVectors.load_word2vec_format(filename, binary=True)
'''

# frWiki2Vec 1.2 billions words
'''# 'femme est pas dans le vocabulaire'
filename = '/home/doudou/Documents/IA_others/we_models/frwiki/frWac_postag_no_phrase_1000_skip_cut100.bin'
fr_model = KeyedVectors.load_word2vec_format(filename, binary=True)
'''

# roi + (femme - homme) = reine
x_word = 'roi'
result = fr_model.most_similar(positive=['femme',x_word], negative=['homme'], topn=1)
print_result(result,x_word)

# ecologie + (mauvais - bon) = pollution
x_word = "écologie"
result = fr_model.most_similar(positive=['mauvais',x_word], negative=['bon'], topn=3)
print_result(result,x_word)

# bon + (pollution - ecologie) = mauvais
x_word = "bon"
result = fr_model.most_similar(positive=['pollution',x_word], negative=['ecologie'], topn=3)
print_result(result,x_word)
# result = mauvais -> donc pollution est mauvais pour ecologie

x_word = "bon"
result = fr_model.most_similar(positive=['oxygène',x_word], negative=['ecologie'], topn=3)
print_result(result,x_word)
