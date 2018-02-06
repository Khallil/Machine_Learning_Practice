
# Split a dataset into a train and test set
from random import seed
from random import randrange

                                # 60 train 40 test
def train_test_split(dataset, split=0.60):
    train = list()
    train_size = split * len(dataset) #0.60 * len(dataset)
    dataset_copy = list(dataset) # copie du dataset
    while len(train) < train_size: 
        index = randrange(len(dataset_copy)) 
        #retourne un nombre dans la range de la longueur du dataset
        train.append(dataset_copy.pop(index))
        #ajoute a la list "train" l'index dataset_copy.pop(4)
        #il semblerait que la fonction list.pop retire l'element de la list
    return train, dataset_copy

#fix the random seed
seed(1)

dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
train, test = train_test_split(dataset)

print(train)
print(test)