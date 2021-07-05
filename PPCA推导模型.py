import numpy as np
import math
import random
from scipy.spatial import distance


class PPCA(object):
    def __init__(self, latent_dim=2, sigma=1.0, max_iter=20):
        # L = 潜在变量的维数
        self.L = latent_dim
        # sigma = 噪声的标准偏差
        self.sigma = sigma
        # D = 数据x的维数
        self.D = 0
        self.data = None
        # N = 数据点数
        self.N = 0
        # mu = 模型的均值
        self.mu = None
        # W = 投影矩阵 DxL
        self.W = None
        # 要做的最大迭代次数
        self.max_iter = max_iter
        self.init = False

    # 可用于标准化数据，但不需要执行PPCA。
    def standarize(self, data):
        if (self.init == False):
            mean = np.mean(data, axis=0)
            self.mu = mean
            data = data - mean
            # 计算标准偏差
            std = np.std(data, axis=0)
            self.std = std
            # 除以标准差
            data /= std
            self.init = True
        else:
            data = data - self.mu
            data /= self.std
        return data

    def inverse_standarize(self, data):
        if (self.init == True):
            data *= self.std
            data = data + self.mu
        return data

    # 拟合模型数据 = W*x + mean + noise_std^2*I
def fit(self, data):
    if (self.init == False):
        mean = np.mean(data, axis=0)
        self.mu = mean
        self.init = True
        self.x = data  # NxD
        self.D = data.shape[1]
        self.N = data.shape[0]
        # mu的封闭形式解决方案是我们现在可以获得的数据的平均值
        # 通过EM算法找到W和sigma ^ 2
        self.expectation_maximization()
    return data

    # 将数据转换为潜在子空间
    def transform_data(self, data):
        W = self.W  # DxL
        sigma = self.sigma
        mu = self.mu  # D dimensions
        M = np.transpose(W).dot(W) + sigma * np.eye(self.L)  # M = W.T*W + sigma^2*I
        Minv = np.linalg.inv(M)  # LxL
        # LxL *     LxD       *         DxN  =   LxN
        latent_data = Minv.dot(np.transpose(W)).dot(np.transpose(data - mu))
        latent_data = np.transpose(latent_data)  # NxL
        return latent_data

    # 将潜在变量转换为原始D维子空间
    def inverse_transform(self, data):  # input is NxL
        #         (DxL   *   L*N).T = NxD    +  Dx1
        # W.dot( np.linalg.inv((W.T).dot(W)) ).dot(M).dot(latent_data.T).T +mu
        M = np.transpose(self.W).dot(self.W) + self.sigma * np.eye(self.L)
        return self.W.dot(np.linalg.inv((self.W.T).dot(self.W))).dot(M).dot(data.T).T + self.mu

    # EM算法找到模型参数W和sigma ^ 2
    def expectation_maximization(self):
        # 将W初始化为较小的随机数
        print("Starting EM algorithm")
        W = np.random.rand(self.D, self.L)
        mu = self.mu
        # 初始sigma为1
        sigma = self.sigma
        L = self.L
        x = self.x
        for i in range(self.max_iter):
            print("iteration " + str(i))
            #         LxD * DxL +   LxL = LxL
            M = np.transpose(W).dot(W) + sigma * np.eye(L)
            Minv = np.linalg.inv(M)
            #        LxL *     LxD          * DxN =    LxN
            ExpZ = Minv.dot(np.transpose(W)).dot((self.x - mu).T)
            #              LxL   +   LxL
            ExpZtrZ = sigma * Minv + ExpZ.dot(np.transpose(ExpZ))  # LxL covariance matrix
            # DxN          NxL      *    LxL  =  DxL
            Wnew = (np.transpose(x - mu).dot(np.transpose(ExpZ))).dot(np.linalg.inv(ExpZtrZ))
            one = np.linalg.norm(x - mu)
            # NxL                 LxD
            two = 2 * np.trace(np.transpose(ExpZ).dot(np.transpose(Wnew)).dot((x - mu).T))
            three = np.trace(ExpZtrZ.dot(np.transpose(Wnew).dot(Wnew)))
            sigmaNew = one - two + three
            sigmaNew = (1 / (self.N * self.D)) * sigmaNew
            sigmaNew = np.absolute(sigmaNew)
            W = Wnew
            sigma = sigmaNew
        self.W = W
        self.sigma = sigma
