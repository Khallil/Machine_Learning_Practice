# -*- coding: utf-8 -*-

#Doudou Khallil
#Gaussian Naive Bayes

import doudou_tools
import numpy
import math

# get le total de (x_i ^ y_i == true)
def get_total_x_class(x_i,x_pos,y_i,dataset):
    # x_i = 1
    # y_i = 1
    count = 0
    for item in dataset:
        if item[x_pos] == x_i and item[-1] == y_i:
            count +=1
    return count

def get_class_prob(dataset):
    row_size = len(dataset[0])
    # we need to calculate the classe probabilities
    c_prob_set = list()
    class_set = [row[-1] for row in dataset]
    classes = set(class_set)
    for c in classes:
        c_prob_set.append([c,float(class_set.count(c)) /float(len(class_set))])
    return c_prob_set
    # -----

def get_mean_and_sstdev_of_set(dataset,pos,y_value):
    x_set = list()
    for item in dataset:
        if item[-1] == y_value:
            x_set.append(item[pos])
    return (doudou_tools.get_mean(x_set),numpy.std(x_set,ddof=1))

def get_mean_sstdev_prob(dataset,classes):
    stat_set = list()
    for y in classes:
        for item in range(len(dataset[0])-1):
            stat_set.append([item,y,get_mean_and_sstdev_of_set(dataset,item,y)])
    return stat_set

def gaussian_probability_density_function(stat_set,new_set,class_prob):
    predict_set = list()
    i = 0
    for new in new_set:
        pdf_set = list()
        map_y = 1
        for stat in stat_set:
            i += 1
            map_y *= 1/(math.sqrt(2*math.pi) * stat[2][1]) * math.exp(-((math.pow(new[stat[0]]-stat[2][0],2) / 2 * math.pow(stat[2][1],2))))            
            if i == len(class_prob):
                map_y *= class_prob[stat[1]][1]
                pdf_set.append([class_prob[stat[1]][0],map_y])
                map_y = 1
                i = 0
        p_set = [row[1] for row in pdf_set]
        index = p_set.index(max(p_set))
        predict_set.append(pdf_set[index][0])
    return predict_set    

dataset = [
[3.393533211,	2.331273381,	0],
[3.110073483,	1.781539638,	0],
[1.343808831,	3.368360954,	0],
[3.582294042,	4.67917911,	    0],
[2.280362439,	2.866990263,	0],
[7.423436942,	4.696522875,	1],
[5.745051997,	3.533989803,	1],
[9.172168622,	2.511101045,	1],
[7.792783481,	3.424088941,	1],
[7.939820817,	0.791637231,	1],]

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

class_set = [row[-1] for row in dataset]
classes = set(class_set)

stat_set = get_mean_sstdev_prob(dataset,classes)
class_set = get_class_prob(dataset)
predict_set = gaussian_probability_density_function(stat_set,testset,class_set)

for item in predict_set:
    print item