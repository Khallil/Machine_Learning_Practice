# -*- coding: utf-8 -*-

#Doudou Khallil
#Bagged Cart

def split_in_sides(model_value,x_n,dataset):
    p_list = list()
    i = 0
    for row in dataset:
        print row[x_n],model_value
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

def get_prediction_list(model_list,dataset):
    i = 0
    p_list = list()
    for model in model_list:
        side_list = split_in_sides(model[0],model[1],dataset)
        v = assign_side_to_class(side_list)
        if i == 0:
            for x in range(len(side_list)):
                if side_list[x][0] == "left":
                    if v == 0:
                        p_list.append([1,0])
                    else:
                        p_list.append([0,1])
                else:
                    if v == 0:
                       p_list.append([0,1])
                    else:
                        p_list.append([1,0])
        else:
            for x in range(len(side_list)):
                if side_list[x][0] == "left":
                    if v == 0:
                        p_list[x][0] += 1
                    else:
                        p_list[x][1] += 1
                else:
                    if v == 0:
                        p_list[x][1] += 1
                    else:
                        p_list[x][0] += 1
        i+=1
    f_list = list()
    for item in p_list:
        if item[0] >= item[1]:
            f_list.append(0)
        else:
            f_list.append(1)
    return f_list

    
model_list= [
    [5.38660215,0],[4.090032824,0],[0.925340325,1]]

dataset = [
[2.309572387,	1.168959634,	0],
[1.500958319,	2.535482186,	0],
[3.107545266,	2.162569456,	0],
[4.090032824,	3.123409313,	0],
[5.38660215,	2.109488166,	0],
[6.451823468,	0.242952387,	1],
[6.633669528,	2.749508563	,1],
[8.749958452,	2.676022211	,1],
[4.589131161,	0.925340325,	1],
[6.619322828,	3.831050828,	1],]

p_list = get_prediction_list(model_list,dataset)
for item in p_list:
    print item