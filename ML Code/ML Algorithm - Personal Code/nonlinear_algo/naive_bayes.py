# -*- coding: utf-8 -*-

#Doudou Khallil
#Naive Bayes

# get le total de (x_i ^ y_i == true)
def get_total_x_class(x_i,x_pos,y_i,dataset):
    # x_i = 1
    # y_i = 1
    count = 0
    for item in dataset:
        if item[x_pos] == x_i and item[-1] == y_i:
            count +=1
    return count

def get_class_and_input_prob(dataset):
    row_size = len(dataset[0])
    # we need to calculate the classe probabilities
    c_prob_set = list()
    class_set = [row[-1] for row in dataset]
    classes = set(class_set)
    for c in classes:
        c_prob_set.append([c,float(class_set.count(c)) /float(len(class_set))])
    # -----
    #we need to calculate the conditional probabilities for each type of input
    i_prob_set = list()
    for x in range(row_size -1):
         x_set = [row[x] for row in dataset]
         x_unique = set(x_set)
         for x_i in x_unique:
             for c in classes:
                i_prob_set.append([x,x_i,c,float(get_total_x_class(x_i,x,c,dataset))/float(class_set.count(c))])
    return c_prob_set,i_prob_set

def get_right_prob(input_i,class_i,pos,prob_set):
    # pour y = 0 je dois recevoir
        # 0 1 0
        # 1 1 0
    # pour y = 1 je dois recevoir
        # 0 1 1
        # 1 1 1

    print "pos : ",pos," i : ",input_i," c : ",class_i
    for item in prob_set:
        if item[0] == pos and item[1] == input_i and item[2] == class_i:
            return item[3]

def get_map(c_prob_set,i_prob_set,new_input):
    # si input = 1,1 
    # on fais toutes les classes
    # pour chaque classe 
        # classe = 0, on get la valeur de weather = 1 et classe 0
        # et weather = 1 et classe = 0
    predict_set = list()
    input_set = [row[0] for row in i_prob_set]
    inputs = set(input_set)
    for item in new_input:
            map_set = list()    
            for c in c_prob_set:
                total = 1                
                for i in range(len(item)):
                    total *= get_right_prob(item[i],c[0],i,i_prob_set)
                total *= c[1]
                print total
                print c[0]
                map_set.append([c[0],total])
            value_only = [row[1] for row in map_set]
            index = value_only.index(max(value_only))
            predict_set.append(map_set[index][0])
             
            # get la plus grande value de map
            # ajoute le y correspondant au predict set
    return predict_set

dataset =[
[1,1,1],
[0,0,1],
[1,1,1],
[1,1,1],
[1,1,1],
[0,0,0],
[0,0,0],
[1,1,0],
[1,0,0],
[0,0,0],]

testset =[
[1,1],
[0,0],
[1,1],
[1,1],
[1,1],
[0,0],
[0,0],
[1,1],
[1,0],
[0,0],]

c_prob_set,i_prob_set = get_class_and_input_prob(dataset)
for item in c_prob_set:
    print item

print " "
for item in i_prob_set:
    print item

predict_set = get_map(c_prob_set,i_prob_set,testset)
for item in predict_set:
    print item
'''
sunny	working	go-out		    1	1	1
rainy	broken	go-out		    0	0	1
sunny	working	go-out		    1	1	1
sunny	working	go-out		    1	1	1
sunny	working	go-out		    1	1	1
rainy	broken	stay-home		0	0	0
rainy	broken	stay-home		0	0	0
sunny	working	stay-home		1	1	0
sunny	broken	stay-home		1	0	0
rainy	broken	stay-home		0	0	0
'''

