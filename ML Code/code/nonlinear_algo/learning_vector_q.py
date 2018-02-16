# -*- coding: utf-8 -*-

#Doudou Khallil
#Learning Vector Quantization

from random import randint,seed
from math import pow,sqrt
import doudou_tools

def get_random_codebook_vectors(dataset):
    codebook_vectors = list()
    classes = [row[-1] for row in dataset]
    class_unique = set(classes)
    n_classe = len(class_unique)
    n_input = len(dataset)
    n_row = len(dataset[0]) - 2
    n_codebook = n_input/n_classe
    while n_codebook % n_classe != 0:
        n_codebook +=1
    n_divide = n_codebook/n_classe
    x = 1
    c = 0
    for i in range(n_codebook):
        codebook_vectors.append([dataset[randint(0,n_input-1)][randint(0,n_row)],dataset[randint(0,n_input-1)][randint(0,n_row)],c])
        if x % n_divide == 0:
            c +=1
        x += 1
    return codebook_vectors

def get_best_bmu(row,codebook_vectors):
    bmu_set = list()
    for codebook in codebook_vectors:
        result = 0
        for i in range(len(row) - 1):
            result += pow(row[i] - codebook[i],2)
        bmu_set.append(sqrt(result))
    index = bmu_set.index(min(bmu_set))
    return index

def edit_codebook_vector(row,codebook_vectors,alpha,index):
    o_class = row[-1]
    n_row = len(codebook_vectors[0]) - 1
    if codebook_vectors[index][-1] == o_class:
        for i in range(n_row):
            codebook_vectors[index][i] = codebook_vectors[index][i] + alpha * (row[i] - codebook_vectors[index][i])
    else:
        for i in range(n_row):
            codebook_vectors[index][i] = codebook_vectors[index][i] - alpha * (row[i] - codebook_vectors[index][i])

def learning(dataset,codebook_vectors,alpha,epoch):
    for i in range(epoch):
        for row in dataset:
            index = get_best_bmu(row,codebook_vectors)
            edit_codebook_vector(row,codebook_vectors,alpha,index)
        alpha = alpha * (1 - ((i+1)/float(epoch)))

def predicting(testset,codebook_vectors):
    predict_set = list()
    for item in testset:
        index = get_best_bmu(item,codebook_vectors)
        predict_set.append(codebook_vectors[index][-1])
    return predict_set

dataset = [
[3.393533211,2.331273381,0],
[3.110073483,1.781539638,0],
[1.343808831,3.368360954,0],
[3.582294042,4.67917911,0],
[2.280362439,2.866990263,0],
[7.423436942,4.696522875,1],
[5.745051997,3.533989803,1],
[9.172168622,2.511101045,1],
[7.792783481,3.424088941,1],
[7.939820817,0.791637231,1]]

seed(79)
codebook_vectors = get_random_codebook_vectors(dataset)
for item in codebook_vectors:
    print item

alpha = 0.7
learning(dataset,codebook_vectors,alpha,182)
'''codebook_vectors = [
[3.582294042,0.791637231,0],
[7.792783481,2.331273381,0],
[7.939820817,2.866990263,1],
[3.393533211,4.67917911,1],]
'''
for item in codebook_vectors:
    print item

testset = [
[3.393533211,	2.331273381],
[3.110073483,	1.781539638],
[1.343808831,	3.368360954],
[3.582294042,	4.67917911],
[2.280362439,	2.866990263],
[7.423436942,	4.696522875],
[5.745051997,	3.533989803],
[9.172168622,	2.511101045],
[7.792783481,	3.424088941],
[7.939820817,	0.791637231],]

predict_set = predicting(testset, codebook_vectors)
y_set = [row[-1] for row in dataset]

for item in predict_set:
    print item

print(doudou_tools.get_accuracy_of_prediction_classification(y_set,predict_set))
