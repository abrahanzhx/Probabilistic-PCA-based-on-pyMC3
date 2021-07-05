import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from PIL import Image
import pymc3 as pm

# 获取Mnist数据集的所有图片和标签
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
imgs = mnist.train.images
labels = mnist.train.labels

# 输出图片，标签的 type和 shape
print(type(imgs))
print(type(labels))
print(imgs.shape)             # (55000, 784)
print(labels.shape)           # (55000,)

# 采集Mnist数据集的小样本
# 取前1000张图片中的前100张数字9
origin_9_imgs = []
for i in range(1000):
      if labels[i] == 9 and len(origin_9_imgs) < 100:
          origin_9_imgs.append(imgs[i])
print(np.array(origin_9_imgs).shape)   # (100, 784)

# 将numpy数组转换为灰度图片
def array_to_img(array):
    array=array*255
    new_img=Image.fromarray(array.astype(np.float32))
    return new_img

# 拼接图片
def comb_imgs(origin_imgs, col, row, each_width, each_height, new_type):
     new_img = Image.new(new_type, (col* each_width, row* each_height))
     for i in range(len(origin_imgs)):
         each_img = array_to_img(np.array(origin_imgs[i]).reshape(each_width, each_width))
         new_img.paste(each_img, ((i % col) * each_width, (i // col) * each_width))
     return new_img




def ppca(data_mat, latent_dim):
    I = np.ones([100,latent_dim])
    mu = np.mean(data_mat,axis=0)  # mu 为数据样本的均值
    # sigma_ture = np.std(data_mat)

    with pm.Model() as ppca_model:      # 定义两个具有正态分布先验
        w = pm.Normal('w', mu=0, sd=I)  # 正态分布
        sigma = pm.Normal('sigma', mu=0, sd=I)  # 正态分布

        # 似然函数
        origin_9_imgs_obs = pm.Normal('origin_9_imgs_obs', mu=mu, sd=np.transpose(w).dot(w) + sigma*I, observed=origin_9_imgs)

        map_estimate = pm.find_MAP(model=ppca_model)
        trace = pm.sample(1000,start=map_estimate)   # MCMC采样
        ## 检查对后验分布的近似是否合理
        # burnin = 100
        # chain = trace[burnin:]
        # pm.traceplot(chain, lines={'sigma': sigma_ture})
        # pm.summary(chain)
        # pm.autocorrplot(chain)

        # 将数据转换为潜在子空间
        W = trace['w']
        std = trace['sigma']
        M = np.transpose(W).dot(W) + std * np.eye(latent_dim)
        Minv = np.linalg.inv(M)
        latent_data = Minv.dot(np.transpose(W)).dot(np.transpose(data_mat - mu))

        return low_d_data_mat,origin_9_imgs_obs

def pca_test(data_mat, top_n_feat):

    num_data, dim = data_mat.shape
    print(num_data)  # 100
    print(dim)  # 784

    mean_vals = data_mat.mean(axis=0)  # shape:(784,)
    mean_removed = data_mat - mean_vals  # shape:(100, 784)
    cov_mat = np.cov(mean_removed, rowvar=0)  # shape：(784, 784)
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))  # 计算特征值和特征向量，shape分别为（784，）和(784, 784)
    eig_val_index = np.argsort(eig_vals)  # 对特征值进行从小到大排序，argsort返回的是索引，即下标
    eig_val_index = eig_val_index[:-(top_n_feat + 1): -1]  # 最大的前top_n_feat个特征的索引
    reg_eig_vects = eig_vects[:, eig_val_index]
    low_d_data_mat = mean_removed * reg_eig_vects  # shape: (100, top_n_feat), top_n_feat最大为特征总数
    recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_vals  # 根据前几个特征向量重构回去的矩阵，shape:(100, 784)

    return low_d_data_mat, recon_mat



# 降维后的结果
low_d_feat_for_9_imgs, recon_mat_for_9_imgs = pca_test(np.array(origin_9_imgs), 2) # 只取最重要的2个特征
print(low_d_feat_for_9_imgs.shape) # (100, 2)
print(recon_mat_for_9_imgs.shape) # (100, 784)
# 观测样本
ten_origin_9_imgs=comb_imgs(origin_9_imgs, 10, 10, 28, 28, 'L')
low_d_img = comb_imgs(recon_mat_for_9_imgs, 10, 10, 28, 28, 'L')

# 展示降维前后的Minst数据样本
result = Image.new(ten_origin_9_imgs.mode,(560,280))
result.paste(ten_origin_9_imgs,(0,0))
result.paste(low_d_img,(280,0))
result.show()
