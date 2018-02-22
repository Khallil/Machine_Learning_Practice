from os import listdir
from nltk.corpus import stopwords
import string
import re
from collections import Counter

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def get_clean_tokens(text):
    tokens = text.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('',w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def process_docs(directory,vocab):
# walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            next
        # create the full path of the file to open
        path = directory + '/' + filename
        # load document
        add_doc_to_vocab(path,vocab)   
        print('Loaded %s' % filename)

def add_doc_to_vocab(filename,vocab):
    text = load_doc(filename)
    tokens = get_clean_tokens(text)
    vocab.update(tokens)

def save_list(lines,filename):
    data = "\n".join(lines)
    file = open(filename,'w')
    file.write(data)
    file.close()


vocab = Counter()
# specify directory to load
process_docs('txt_sentoken/neg',vocab)
process_docs('txt_sentoken/pos',vocab)
#print(len(vocab))
#print(vocab.most_common(50))
# The counter is mainly here to help us manipulate
# the big vocabulary we get, see the reduction below for example

# keep tokens with > 5 occurrence
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print len(tokens)
save_list(tokens, 'vocab.txt')

# ici on save tout pos+neg dans le meme vocab

