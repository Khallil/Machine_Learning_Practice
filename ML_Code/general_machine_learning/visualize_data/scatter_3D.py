from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
'''for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    print xs
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)

dataset = [
[3.393533211,2.331273381,1,1],
[3.110073483,1.781539638,1,1],
[1.343808831,3.368360954,1,1],
[3.582294042,4.67917911,1,1],
[7.423436942,4.696522875,2,2],
[7.444542326,0.476683375,2,2],
[10.12493903,3.234550982,2,2],
[6.642287351,3.319983761,2,2]]
'''
dataset = [
[5.4189696861,	3.3828798825,	6.0418452372,	-1],
[9.1980603504,	4.3855236444,	10.0537847139,	-1],
[20.1193996931,	5.7587404818,	13.6657987267,	-1],
[13.5778633972,	5.3244568503,	14.7982316337,	-1],
[5.2146397107	,2.9092704443,	3.4344095315,	-1],
[60.9573869605,	7.1676741463,	10.8249694723,	1],
[37.6136661362,	5.1240744379,	4.5820103294,	1],
[56.4726604001,	5.6274636429,	4.4396890369,	1],
[30.2762411174,	3.9307411375,	1.9712233233,	1],
[55.2484835427,	7.9356953831,	17.9456668785,	1]]

xs = [row[0] for row in dataset]
ys = [row[1] for row in dataset]
zs = [row[2] for row in dataset]
classes = [row[3] for row in dataset]
ax.scatter(xs, ys, zs,c=classes)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()