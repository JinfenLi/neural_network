# -*- coding: UTF-8 -*-

"""
network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes,funcntion_piece):
        """
        :param sizes: sizes = [2, 3, 2] means 2d-3d-2d network
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        np.random.seed(0)

        self.biases = [np.random.randn(1,y) for y in sizes[1:]]
        # randomly initiate weight 值（0 - 1）
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
        self.n = funcntion_piece


    def feedforward(self, a):
        """

        :param a
        :return: activate value
        """
        for b, w in zip(self.biases, self.weights):
            # 加权求和以及加上 biase
            a = self.sigmoid(np.dot(a,w) + b)
        return a

    def BGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        随机梯度下降
        :param training_data
        :param epochs
        :param mini_batch_size
        :param eta: learning rate
        :param test_data
        """
        # if test_data:
        n_test = len(test_data)
        n = len(training_data)
        before_loss=1000000
        for j in range(epochs):
            # if the training data is in order, then the result is hard to be converge
            random.seed(60)
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            print("Epoch {0}".format(j))
            total=0
            total_loss=0
            count_batch=0
            for mini_batch in mini_batches:
                # calculate MSE each batch
                total+=len(mini_batch)
                loss=self.update_mini_batch(mini_batch, eta)
                # print("{0}/{1}-----------------{2}".format(total,n_test,loss/mini_batch_size))
                total_loss+=loss
                count_batch+=1
            # print MSE after every epoch
            if len(test_data)>0:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
                print("Epoch {0}: {1} ".format(
                    j, total_loss/count_batch))
                print("**********************************************")
            else:
                print ("Epoch {0} complete".format(j))
            # stop criteria
            if total_loss>=before_loss and j>10:
                break
            before_loss=total_loss

    def update_mini_batch(self, mini_batch, eta):
        """
        update w, b
        :param mini_batch
        :param eta: learning rate
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        loss=0
        for x in mini_batch:
            # calculate the delta of w and b and MSE
            delta_nabla_b, delta_nabla_w,loss = self.backprop(x[:-1], x[-1],loss)
            # aggregate delta_nabla_b 和 delta_nabla_w
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # normalized by the batch size
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        return loss

    def backprop(self, x, y,loss):
        """
        :param x:
        :param y:
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        # store the activated value
        activations = [np.array([x])]
        # store the unactivated value
        zs = []
        for b, w in zip(self.biases, self.weights):

            z = np.dot(activation,w)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        #  δ =-(target-out)*net_o*(1-net_o)*out_h1

        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        # print(np.square(self.cost_derivative(activations[-1], y)))
        loss+= np.square(self.cost_derivative(activations[-1], y))
        # print(loss)
        nabla_b[-1] = delta

        nabla_w[-1] = np.dot(activations[-2].T,delta)
        for l in range(2, self.num_layers):

            z = zs[-l]
            sp = self.sigmoid_prime(z)
            # delta1=delta0*w2*sp

            delta = np.sum(np.dot(delta,self.weights[-l+1].transpose()))*sp

            nabla_b[-l] = delta

            nabla_w[-l] = np.dot(activations[-l-1].T,delta)
        return (nabla_b, nabla_w,loss)

    def evaluate(self, test_data):

        # print(test_data)
        x = test_data[:,:-1]
        y = test_data[:,-1]
        # print(x)
        # print(y)
        test_results = [1 if z[0]>0.5 else 0 for z in self.feedforward(x)]
        # print(test_results)
        # return the number of right label
        return sum(int(x == y) for (x, y) in zip(test_results,y))

    def cost_derivative(self, output_activations, y):
        """
        :param output_activations:
        :param y:
        :return:
        """
        # print(" {0} / {1}************************{2}".format(i, n_test,np.square(output_activations[0][0]-y)))
        # print(output_activations[0][0])
        return (output_activations[0][0]-y)


    # #### Miscellaneous functions
    def sigmoid(self,z):
        """

        :param z:
        :return:
        """
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self,z):
        """
        derivative of sigmoid
        :param z:
        :return:
        """
        return self.sigmoid(z)*(1-self.sigmoid(z))

    # def flinear(self,z,n):
    #     result=0
    #     if n==4:
    #         # n=4
    #         if z <= -2.5:
    #             result = 0.0725
    #         elif -2.5 < z <=  0:
    #             result = 0.170 * z + 0.4975
    #
    #         elif 0 < z <=  2.5:
    #             result = 0.171 * z + 0.4975
    #         else:
    #             result = 1
    #     elif n==6:
    #
    #         # n=6
    #         if z <= -2.5:
    #             result = 0.0725
    #         elif -2.5 < z <= -2.0:
    #             result = 0.087 * z + 0.29
    #         elif -2.0 < z <=  0:
    #             result = 0.190 * z + 0.496
    #
    #         elif 0 < z <= 2.0:
    #             result = 0.191 * z + 0.5405
    #         elif 2.0 < z <= 2.5:
    #             result = 0.245 * z + 0.4325
    #         else:
    #             result = 1
    #     elif n==8:
    #         #n=8
    #         if z <= -2.5:
    #             result = 0.0725
    #         elif -2.5 < z <= -2.0:
    #             result = 0.087 * z + 0.29
    #         elif -2.0 < z <= -1.5:
    #             result = 0.126 * z + 0.368
    #
    #         elif -1.5 < z <= 0:
    #             result = 0.212 * z + 0.497
    #
    #         elif 0 < z <= 1.5:
    #             result = 0.230 * z + 0.469
    #         elif 1.5 < z <= 2.0:
    #             result = 0.217 * z + 0.4885
    #         elif 2.0 < z <= 2.5:
    #             result = 0.245 * z + 0.4325
    #         else:
    #             result = 1
    #     elif n==10:
    #         # n=10
    #         if z <= -2.5:
    #             result = 0.0725
    #         elif -2.5 < z <= -2.0:
    #             result = 0.087 * z + 0.29
    #         elif -2.0 < z <= -1.5:
    #             result = 0.126 * z + 0.368
    #
    #         elif -1.5 < z <= -1.0:
    #             result = 0.173 * z + 0.4385
    #         elif -1.0 < z <= 0:
    #             result = 0.231 * z + 0.4965
    #
    #         elif 0 < z <= 1.0:
    #             result = 0.230 * z + 0.4975
    #
    #
    #         elif 1.0 < z <= 1.5:
    #             result = 0.173 * z + 0.5545
    #         elif 1.5 < z <= 2.0:
    #             result = 0.217 * z + 0.4885
    #         elif 2.0 < z <= 2.5:
    #             result = 0.245 * z + 0.4325
    #         else:
    #             result = 1
    #
    #     elif n==12:
    #         # n=12
    #         if z<=-2.5:
    #             result=0.0725
    #         elif -2.5<z<=-2.0:
    #             result=0.087*z+0.29
    #         elif -2.0<z<=-1.5:
    #             result=0.126*z+0.368
    #
    #         elif -1.5<z<=-1.0:
    #             result=0.173*z+0.4385
    #         elif -1.0<z<=-0.5:
    #             result=0.217*z+0.4825
    #         elif -0.5<=0:
    #             result=0.245*z+0.4965
    #         elif 0<z<=0.5:
    #             result=0.244*z+0.4965
    #         elif 0.5<z<=1.0:
    #             result=0.218*z+0.5095
    #
    #         elif 1.0<z<=1.5:
    #             result=0.173*z+0.5545
    #         elif 1.5<z<=2.0:
    #             result=0.217*z+0.4885
    #         elif 2.0<z<=2.5:
    #             result=0.245*z+0.4325
    #         else:
    #             result=1
    #     return result
    #
    # def flinear_prime(self,z,n):
    #     result=0
    #     if n==4:
    #         # n=4
    #         if z <= -2.5:
    #             result = 0
    #         elif -2.5 < z <= 0:
    #             result = 0.170
    #
    #         elif 0 < z <= 2.5:
    #             result = 0.171
    #         else:
    #             result = 0
    #     elif n==6:
    #
    #         # n=6
    #         if z <= -2.5:
    #             result = 0.0725
    #         elif -2.5 < z <= -2.0:
    #             result = 0.087
    #         elif -2.0 < z <=  0:
    #             result = 0.190
    #
    #         elif 0 < z <= 2.0:
    #             result = 0.191
    #         elif 2.0 < z <= 2.5:
    #             result = 0.245
    #         else:
    #             result = 0
    #     elif n==8:
    #         # n=8
    #         if z <= -2.5:
    #             result = 0
    #         elif -2.5 < z <= -2.0:
    #             result = 0.087
    #         elif -2.0 < z <= -1.5:
    #             result = 0.126
    #
    #         elif -1.5 < z <= 0:
    #             result = 0.212
    #
    #         elif 0 < z <= 1.5:
    #             result = 0.230
    #         elif 1.5 < z <= 2.0:
    #             result = 0.217
    #         elif 2.0 < z <= 2.5:
    #             result = 0.245
    #         else:
    #             result = 0
    #
    #     elif n==10:
    #         # n=10
    #         if z <= -2.5:
    #             result = 0
    #         elif -2.5 < z <= -2.0:
    #             result = 0.087
    #         elif -2.0 < z <= -1.5:
    #             result = 0.126
    #
    #         elif -1.5 < z <= -1.0:
    #             result = 0.173
    #         elif -1.0 < z <=  0:
    #             result = 0.231
    #         elif 0 < z <= 1.0:
    #             result = 0.230
    #
    #         elif 1.0 < z <= 1.5:
    #             result = 0.173
    #         elif 1.5 < z <= 2.0:
    #             result = 0.217
    #         elif 2.0 < z <= 2.5:
    #             result = 0.245
    #         else:
    #             result = 0
    #
    #     elif n==12:
    #         # n=12
    #         if z<=-2.5:
    #             result=0
    #         elif -2.5<z<=-2.0:
    #             result=0.087
    #         elif -2.0<z<=-1.5:
    #             result=0.126
    #
    #         elif -1.5<z<=-1.0:
    #             result=0.173
    #         elif -1.0<z<=-0.5:
    #             result=0.217
    #         elif -0.5<=0:
    #             result=0.245
    #         elif 0<z<=0.5:
    #             result=0.244
    #         elif 0.5<z<=1.0:
    #             result=0.218
    #
    #         elif 1.0<z<=1.5:
    #             result=0.173
    #         elif 1.5<z<=2.0:
    #             result=0.217
    #         elif 2.0<z<=2.5:
    #             result=0.245
    #         else:
    #             result=0
    #     return result
    #
    # def sigmoid(self,z):
    #     return np.array([list(map(lambda x:self.flinear(x,self.n),z[0]))])
    #
    #
    #
    #
    # def sigmoid_prime(self,z):
    #     return np.array([list(map(lambda x: self.flinear_prime(x,self.n), z[0]))])

def main():
    import time


    import pandas as pd
    df=pd.read_csv('heart.csv').values

    X = np.array(df)


    nn = Network([13, 26, 1],funcntion_piece=6)
    starttime = time.time()
    nn.BGD(training_data=X,epochs=50,mini_batch_size=16,eta=0.5,test_data=X)
    endtime = time.time()
    dtime = endtime - starttime

    print("used time：%.8s s" % dtime)



if __name__ == "__main__":
    main()
