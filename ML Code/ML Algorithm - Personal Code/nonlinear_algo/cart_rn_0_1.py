# -*- coding: utf-8 -*-

#Doudou Khallil
#Classification Tree from Scratch

# départage les valeurs en left ou right
def split_x_in_side(n,set_x,p):
    left = [0,0]
    right = [0,0]
    for x in set_x:
        if x[p] < n:
            if x[-1] == 0:
                left[0] += 1
            else:
                left[1] += 1
        else:
            if x[-1] == 0:
                right[0] += 1
            else:
                right[1] += 1
    return left,right

def calculate_gini_index(left_0,left_1,right_0,right_1):
    gini = (left_0 * (1- left_0)) + (right_0 * (1 - right_0)) + (left_1 * (1 - left_1)) + (right_1 * (1 - right_1))
    return gini

# calculate la proportion d'appartenance des left,right en classe 0 ou 1
def calculate_proportion(left, right):
    left_sum = left[0] + left[1]
    right_sum = right[0] + right[1]
    if left_sum > 0:
       left[0] = left[0] / float(left_sum)
       left[1] = left[1] / float(left_sum)
    else:
        return False
    if right_sum > 0:
        right[0] = right[0] / float(right_sum)
        right[1] = right[1] / float(right_sum)
    else:
        return False

# retourne la valeur x du point à partir du gini index
def return_split_x_point(gini_set):
    print "RETURN SPLIT X POINT"
    for item in gini_set:
        print " ",item
    gini_only = [row[0] for row in gini_set]
    index = gini_only.index(min(gini_only))
    return gini_set[index][1],gini_set[index][2],gini_set[index][3]

def split_dataset_based_on_x_value(dataset,_x,n):
    left_set = list()
    right_set = list()
    for x in dataset:
        if x[n] < _x:
            left_set.append(x)
        else:
            right_set.append(x)
    return left_set,right_set

#check le contenu du dataset pour savoir si il est déja uniforme
def check_uniformity(s_set,l):
    if l == "left":
        print "\nCHECK UNIFORMITY LEFT"
    if l == "right":
        print "\nCHECK UNIFORMITY RIGHT"

    print " s_set",s_set
    p = s_set[0][-1]
    for item in s_set:
        if item[-1] != p:
            return False
    return True

def update_final_set(final_set,n_set,depth,terminal):
    # chercher le x dans le set, et update son
    # gauche, droite en fonction des parametres

    print "\nUPDATE_final_set"
    print " final_set : ",final_set
    print " n_set : ",n_set
    print " depth : ",depth
    print " terminal : ",terminal
    n_set.append(depth)
    n_set.append(terminal)
    if len(final_set) == 0:
        final_set.append(n_set)
    else:
        final_set.append(n_set)
    print " after append : ", final_set

# lance l'apprentissage
def finding_best_split_point(dataset,final_set,depth):
    print "\nFINDING Best Split Point"
    print " depth : ",depth
    left = list()
    right = list()
    gini_set = list()
    
    row = dataset[0]
    p = len(row) - 1
    for i in range(p):
        for item in dataset:
            left,right = split_x_in_side(item[i],dataset,i)
            if calculate_proportion(left,right) != False:
                gini = calculate_gini_index(left[0],left[1],right[0],right[1])
                if gini == 0.0:
                    print "!!! gini = 0.0"
                    update_final_set(final_set,[item[i],i,item[-1]],depth,"final")
                    return 0
                gini_set.append([gini,item[i],i,item[-1]])
    x,n,y = return_split_x_point(gini_set)
    print " best split point : ", x,n,y
    left_set,right_set = split_dataset_based_on_x_value(dataset,x,n)

    isUniform = check_uniformity(left_set,"left")    
    if isUniform:
        # c'est une terminal à gauche
        update_final_set(final_set,[x,n,y],depth,"left")
    else:
        # c'est pas un terminal
        finding_best_split_point(left_set,final_set,depth + 1)
        #update_final_set(x,n,y,depth)
    
    isUniform = check_uniformity(right_set,"right")    
    if isUniform:
        update_final_set(final_set,[x,n,y],depth,"right")
    else:
        finding_best_split_point(right_set,final_set,depth + 1)
        #update_final_set(x,n,y,depth)

# fais la prédiction avec le split_point
def cart(final_set,input,i):
    pos = final_set[i][1]
    if input[pos] < final_set[i][pos]:
        if final_set[i][-1] == "left" or final_set[i][-1] == "final":
            if final_set[i][2] == 0:
                return 1
            if final_set[i][2] == 1:
                return 0
        else:
            return cart(final_set,input,i+1)
    if input[pos] >= final_set[i][pos]:
        if final_set[i][-1] == "right" or final_set[i][-1] == "final":
            return final_set[i][2]
        else:
            return cart(final_set,input,i+1)

dataset = [
[2.771244718,1.784783929,0],
[1.728571309,1.169761413,0],
[2.999208922,3.234255098,0],
[2.339208922,2.664255098,0],
[3.199208922,1.934255098,1],
[2.399208922,1.534255098,1],
[2.497545867,3.162953546,1],
[2.12493903,2.209014212,1],
[1.642287351,3.319983761,1]]

final_set = list()
finding_best_split_point(dataset,final_set,1)
print "\nFINAL_SET"
print "value, Rn, class, depth"

final_set = sorted(final_set,key=lambda depth:depth[3])

for item in final_set:
    print item

_input = [3.12,2.19]
print cart(final_set,_input,0)
