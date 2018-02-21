# -*- coding: utf-8 -*-

#Doudou Khallil
#K-Nearest Neighbors

from math import pow,sqrt
from collections import Counter

def get_smaller_one(k,result_set):
    class_values = list()
    values_column = [row[0] for row in result_set]
    class_column = [row[1] for row in result_set]
    while k > 0:
        r = values_column.index(min(values_column))
        class_values.append(class_column[r])
        del values_column[r]
        del class_column[r]
        k -= 1
    count = Counter(class_values)
    return count.most_common()[0][0]

def get_euclidean_distance(x,y):
    r = 0
    for i in range(len(x)-1):
        r += pow(x[i] - y[i],2)
    return sqrt(r)

def knearest_neighbors(dataset,testset):
    predicted_set = []
    result_set = []
    for item in testset:
        for d_item in dataset:
            result_set.append([get_euclidean_distance(item,d_item),d_item[-1]])
        predicted_set.append(get_smaller_one(3,result_set))
        result_set = []
    return predicted_set

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


testset = [
[2.771244718,1.784783929,0],
[1.728571309,1.169761413,0],
[3.678319846,2.8128135,0],
[3.961043357,2.61995032,0],
[2.999208922,2.209014212,0],
[7.497545867,3.162953546,1],
[9.00220326,3.339047188,1],
[7.444542326,0.476683375,1],
[10.12493903,3.234550982,1],
[6.642287351,3.319983761,1]]

print(knearest_neighbors(dataset,testset))