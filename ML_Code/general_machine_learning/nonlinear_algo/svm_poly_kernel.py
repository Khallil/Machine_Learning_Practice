from math import pow,sqrt
import doudou_tools
#1 z1 =  x1^2 , z2 =   SQRT(2 * x1 * x2) , z3 = x2^2
def convert_r2_to_r3_set(dataset):
    r3_set = list()
    for item in dataset:
        r3_set.append([pow(item[0],2),sqrt(2*item[0]*item[1]),pow(item[1],2),item[-1] ])
    return r3_set

def multivariate_gradient_descent(dataset, alpha, n_epoch):
    coef_set = [0.0 for i in range(len(dataset[0]))]
    print coef_set
    for i in range(n_epoch):
        for x in range(len(dataset)):
            p_y = predict(coef_set,dataset[x])
            error = p_y - dataset[x][-1]
            coef_set[0] = coef_set[0] - alpha * error
            for i in range(len(coef_set)-1):
                coef_set[i+1] = coef_set[i+1] - alpha * error * dataset[x][i]
        n_epoch -= 1
    return coef_set

# predict function (row,y)
def predict(coef_set,row):
    sum = coef_set[0]
    for x in range(len(row)-1):
        sum += coef_set[x+1] * row[x]
    return sum

def predict_set(coef_set,dataset):
    predicted_set = list()
    for row in dataset:
        if predict(coef_set,row) >= 0.5:
            predicted_set.append(1)
        else:
            predicted_set.append(0)
    return predicted_set

dataset = [
[2.327868056,2.458016525,	-1],
[3.032830419,	3.170770366,	-1],
[4.485465382,	3.696728111	,-1],
[3.684815246,	3.846846973,	-1],
[2.283558563,	1.853215997	,-1],
[7.807521179,	3.290132136,	1],
[6.132998136,	2.140563087	,1],
[7.514829366,	2.107056961,	1],
[5.502385039,	1.404002608	,1],
[7.432932365,	4.236232628,	1],]


alpha = 0.1
n_epoch = 30

r3_set = convert_r2_to_r3_set(dataset)
r3_set = doudou_tools.normalize_dataset_with_y(r3_set)
for item in r3_set:
    print item

coef_set = multivariate_gradient_descent(r3_set,alpha,n_epoch)
for item in coef_set:
    print item

p_set = predict_set(coef_set,r3_set)
for item in p_set:
    print item