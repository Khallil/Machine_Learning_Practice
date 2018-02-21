
from keras.preprocessing.text import Tokenizer

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def get_split_train_test(docs,labels,value):
    i = 0
    train_docs = list()
    test_docs = list()
    y_train = list()
    y_test = list()
    for item in docs:
        if i <= value:
            train_docs.append(docs[i])
            y_train.append(labels[i])
        else:
            test_docs.append(docs[i])
            y_test.append(labels[i])
        i+=1
    return train_docs,y_train,test_docs,y_test

text = load_doc("negative.txt")
tokens_neg = text.split()
text = load_doc("positive.txt")
tokens_pos = text.split()

labels = [0 for _ in range(len(tokens_neg))] + [1 for _ in range(len(tokens_pos))]
tokens = tokens_neg + tokens_pos
# pour un split 80/20 train/test
train_v = len(tokens) * 0.8
train_docs,ytrain,test_docs,ytest = get_split_train_test(tokens,labels,train_v)

tokenizer = create_tokenizer(train_docs)
Xtrain = tokenizer.texts_to_matrix(train_docs, mode='freq')
Xtest = tokenizer.texts_to_matrix(test_docs, mode='freq')
print (Xtrain.shape, Xtest.shape)