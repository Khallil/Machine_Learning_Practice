#-*- coding: utf-8 -*-

#Doudou Khallil
#Adaboost

from math import log,exp

def split_in_sides(model_value,x_n,dataset):
    p_list = list()
    i = 0
    for row in dataset:
        if row[x_n] <= model_value:
            p_list.append(["left",row[-1]])
        else:
            p_list.append(["right",row[-1]])
        i+=1
    return p_list

def assign_side_to_class(side_list):
    # on prends la list 
    left_0 = 0
    left_1 = 0
    for item in side_list:
        if item[0] == "left" and item[1] == 0:
            left_0 +=1
        elif item[0] == "left" and item[1] == 1:
            left_1 +=1
    if left_0 >= left_1:
        return 0
    else:
        return 1

def update_weight_list(weight_list,weight_error,stage):
    for x in range(len(weight_list)):
        weight_list[x] = exp(stage*weight_error[x])

def learning(dataset,model_list,alpha,y_set):
    weight_list = list()
    p_list_set = list()
    stage_set = list()
    for x in range(len(dataset)):
        weight_list.append(alpha)
    for model in model_list:
        p_list = list()
        weight_error = list()
        side_list = split_in_sides(model[0],model[1],dataset)
        v = assign_side_to_class(side_list)
        for item in side_list:
            if item[0] == "left":
                p_list.append(v)
            else:
                p_list.append(1-v)
        for i in range(len(p_list)):
            if p_list[i] != y_set[i]:
                weight_error.append(weight_list[i])
            else:
                weight_error.append(0)
        m_rate = float(sum(weight_error))/float(sum(weight_list))
        stage = log((1-m_rate)/m_rate)
        p_list_set.append([p_list,stage])        
        update_weight_list(weight_list,weight_error,stage)
    return p_list_set,stage_set

def prediction(p_list_set):
    model_values_set = list()
    for item in p_list_set:
        model_values = list()
        for i in range(len(item[0])):
            if item[0][i] == 0:
                model_values.append(item[1] * -1)
            else:
                model_values.append(item[1] * 1)
        model_values_set.append(model_values)
    y_set = list()
    for i in range(len(model_values_set[0])):
        r = 0
        for x in range(len(p_list_set)):
            r += model_values_set[x][i]
        if r < 0:
            y_set.append(0)   
        else:
            y_set.append(1)
    return y_set

model_list = [[4.932600453,0],[2.122873405,1],[0.862698005,1],]

alpha = 0.1

dataset = [
[3.64754035,	2.996793259	,0],
[2.612663842,	4.459457779	,0],
[2.363359679,	1.506982189	,0],
[4.932600453,	1.299008795	,0],
[3.776154753,	3.157451378	,0],
[8.673960793,	2.122873405	,1],
[5.861599451,	0.003512817	,1],
[8.984677361,	1.768161009	,1],
[7.467380954	,0.187045945,1],
[4.436284412,	0.862698005,1],]

y_set = [row[-1] for row in dataset]
p_list_set, stage_set  = learning(dataset,model_list,alpha,y_set)
y_set = prediction(p_list_set)
for item in y_set:
    print item

