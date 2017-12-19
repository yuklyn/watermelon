import FNN
import matplotlib.pyplot as plt
import numpy as np

x = np.mat('2,3,3,2,1,2,3,3,3,2,1,1,2,1,3,1,2;\
        1,1,1,1,1,2,2,2,2,3,3,1,2,2,2,1,1;\
        2,3,2,3,2,2,2,2,3,1,1,2,2,3,2,2,3;\
        3,3,3,3,3,3,2,3,2,3,1,1,2,2,3,1,2;\
        1,1,1,1,1,2,2,2,2,3,3,3,1,1,2,3,2;\
        1,1,1,1,1,2,2,1,1,2,1,2,1,1,2,1,1;\
        0.697,0.774,0.634,0.668,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719;\
        0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103\
        ').T
x = np.array(x)
y = np.mat('1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0')
y = np.array(y).T
xx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

sampleNum, featuresNum = x.shape
np.random.seed(0)

print('样本数: ', sampleNum)
print('特征数: ', featuresNum)

fnn = FNN.FNN()
fnn.create(featuresNum, featuresNum + 1, 1)

e = []
for i in range(2000):
    err, err_k = fnn.train_bp_standard(x, y.reshape(len(y), 1), 0.5)
    e.append(err)

plt.figure(2)
plt.xlabel("epochs")
plt.ylabel("accumulated error")
plt.title("convergence curve")
plt.plot(e)
plt.show()

predict_y = fnn.do_predict(x)

plt.figure(figsize=(8, 4))
plt.xlabel("sample order")
plt.ylabel("classification")
plt.title("watermelon classification")
plt.plot(xx, y, "b--", linewidth=1)
plt.plot(xx, predict_y, "r--", linewidth=1)
plt.show()
