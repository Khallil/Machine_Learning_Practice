# -*- coding: utf-8 -*-

# Khallil Doudou
# SVM SubGradientDescent
# Classification Binaire

import doudou_tools

def get_output(x1,x2,b1,b2,y):
    return y * (b1 * x1) + (b2 * x2)

def learning(dataset,b1,b2,lbda,epoch):
    t = 1
    for i in range(epoch):
        for item in dataset:
            output = get_output(item[0],item[1],b1,b2,item[2])
            if output < 1:
                b1 = (1 - 1/t) * b1 + 1/(lbda * t) * (item[-1] * item[0])
                b2 = (1 - 1/t) * b2 + 1/(lbda * t) * (item[-1] * item[1])
            if output >= 1:
                b1 = (1 - 1/t) * b1
                b2 = (1 - 1/t) * b2
            t +=1
    return b1,b2

def predict(testset,b1,b2):
    predict_set = list()
    for item in testset:
        result = (b1 * item[0]) + (b2 * item[1])
        if result < 0:
            predict_set.append(-1)
        else:
            predict_set.append(1)
    return predict_set

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

testset = [
[2.327868056,2.458016525],
[3.032830419,	3.170770366],
[4.485465382,	3.696728111],
[3.684815246,	3.846846973],
[2.283558563,	1.853215997],
[7.807521179,	3.290132136],
[6.132998136,	2.140563087],
[7.514829366,	2.107056961],
[5.502385039,	1.404002608],
[7.432932365,	4.236232628]]

b1 = 0
b2 = 0
lbda = 0.45
epoch = 300
b1,b2 = learning(dataset,b1,b2,lbda,epoch)
y_set = [row[-1] for row in dataset]
predict_set = predict(testset,b1,b2)

print doudou_tools.get_accuracy_of_prediction_classification(y_set,predict_set)