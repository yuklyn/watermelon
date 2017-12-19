class FNN:
    def __init__(self):
        self.i_num = 0
        self.h_num = 0
        self.o_num = 0

        self.h_threshold = []
        self.o_threshold = []

        self.i_h_weight = []
        self.h_o_weight = []

        self.i_output = []
        self.h_output = []
        self.o_output = []

    def create(self, in_num, hide_num, out_num):
        import numpy as np

        self.i_num = in_num
        self.h_num = hide_num
        self.o_num = out_num

        self.h_threshold = np.random.random(hide_num)
        self.o_threshold = np.random.random(out_num)

        self.i_h_weight = np.random.random((self.i_num, self.h_num))
        self.h_o_weight = np.random.random((self.h_num, self.o_num))

        self.i_output = np.zeros(in_num)
        self.h_output = np.zeros(hide_num)
        self.o_output = np.zeros(out_num)

    def prediction(self, x):
        for i in range(self.i_num):
            self.i_output[i] = x[i]

        for h in range(self.h_num):
            temp = 0.0
            for i in range(self.i_num):
                temp += self.i_output[i] * self.i_h_weight[i][h]
            self.h_output[h] = self.sigmoid(temp - self.h_threshold[h])

        for j in range(self.o_num):
            temp = 0.0
            for h in range(self.h_num):
                temp += self.h_output[h] * self.h_o_weight[h][j]
            self.o_output[j] = self.sigmoid(temp - self.o_threshold[j])

    def back_propaganda(self, x, y, learn_rate):
        import numpy as np
        self.prediction(x)

        g = np.zeros(self.o_num)
        for j in range(self.o_num):
            g[j] = self.o_output[j] * (1 - self.o_output[j]) * (y[j] - self.o_output[j])

        e = np.zeros(self.h_num)
        for h in range(self.h_num):
            temp = 0
            for j in range(self.o_num):
                temp += g[j] * self.h_o_weight[h][j]
            e[h] = self.h_output[h] * (1 - self.h_output[h]) * temp

        for j in range(self.o_num):
            self.o_threshold[j] += - (learn_rate * g[j])

        for h in range(self.h_num):
            for j in range(self.o_num):
                self.h_o_weight[h][j] += learn_rate * g[j] * self.h_output[h]

        for h in range(self.h_num):
            self.h_threshold[h] += - (learn_rate * e[h])

        for i in range(self.i_num):
            for h in range(self.h_num):
                self.i_h_weight[i][h] += learn_rate * e[h] * self.i_output[i]

    def train_bp_standard(self, data_x, data_y, learn_rate=0.05):
        errors = []

        for k in range(len(data_x)):
            self.back_propaganda(data_x[k], data_y[k], learn_rate)
            error = 0
            for j in range(self.o_num):
                error += (data_y[k] - self.o_output[j]) * (data_y[k] - self.o_output[j])
            errors.append(error / 2)

        e = sum(errors) / len(errors)

        return e, errors

    def do_predict(self, data_x):
        predict_outputs = []
        for k in range(len(data_x)):
            self.prediction(data_x[k])
            if self.o_output[0] > 0.5:
                predict_outputs.append(1)
            else:
                predict_outputs.append(0)
        return predict_outputs

    @staticmethod
    def sigmoid(x):
        from math import exp
        return 1.0 / (1.0 + exp(-x))
