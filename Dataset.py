import scipy.sparse as sp
import numpy as np
import pandas as pd
import os
from gensim.models.doc2vec import Doc2Vec
import logging


class Dataset(object):
    '''
    数据集类
    '''

    def __init__(self, path):
        '''
        构造函数
        参数: path - 包含训练和测试csv文件以及特征数据的路径
        '''
        self.logger = logging.getLogger('Dataset')
        self.logger.setLevel(logging.DEBUG)
        # 读取训练.csv文件并将其转换为稀疏矩阵对象trainMatrix和评级数量train_num
        self.trainMatrix, self.train_num = self.load_rating_file_as_matrix(path + "/train.csv")
        # 读取测试.csv文件并将其转换为稀疏矩阵对象testRatings和评级数量test_num
        self.testRatings, self.test_num = self.load_rating_file_as_matrix(path + "/test.csv")
        # 加载文本和图像特征并将它们存储在类属性textualfeatures和imagefeatures中
        self.textualfeatures, self.imagefeatures, = self.load_features(path)
        # 获取trainMatrix的shape属性，分别存储num_users和num_items到类属性num_users和num_items中
        self.num_users, self.num_items = self.trainMatrix.shape
        # 使用 logger 输出数据集相关信息
        self.logger.info(f"Loaded dataset with {self.num_users} users and {self.num_items} items.")
        self.logger.info(
            f"Training set contains {self.train_num} ratings and test set contains {self.test_num} ratings.")

    def load_rating_file_as_matrix(self, filename):
        '''
        读取 .csv 文件并将其转换为稀疏矩阵对象
        .csv文件的第一行格式应该为 num_users\t num_items
        参数: filename - 要读取的csv文件名
        返回值: 稀疏矩阵对象和评级数量
        '''
        # 获取用户数、项目数和总评级数
        num_users, num_items, num_total = 0, 0, 0
        df = pd.read_csv(filename, index_col=None, usecols=None)
        # 迭代处理每一行数据
        for index, row in df.iterrows():
            # 提取该行数据中的userID和itemID字段，并将其转换为整数类型
            u, i = int(row['encoded_userID']), int(row['encoded_itemID'])
            # 更新用户数/项目数为当前最大值
            num_users = max(num_users, u)
            num_items = max(num_items, i)

        # 构造稀疏矩阵对象
        # 按照用户数和项目数创建一个空的稀疏矩阵，用于存储评级信息
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        for index, row in df.iterrows():
            # 提取该行数据中的userID和itemID字段，并将其转换为整数类型，同时设定当前评级为1.0
            user, item, rating = int(row['encoded_userID']), int(row['encoded_itemID']), 1.0
            # 若当前评级大于0，则在稀疏矩阵中标记该项为1.0
            if (rating > 0):
                mat[user, item] = 1.0
                # 更新总评级数
                num_total += 1

        # 返回稀疏矩阵和评级数量
        self.logger.info(f"Loaded {filename} with {num_users+1} users and {num_items+1} items.")
        self.logger.info(f"Matrix size: {mat.shape}, total ratings: {num_total}")
        return mat, num_total

    def load_features(self, data_path):

        # Prepare textual feature data.
        # npz存储单个Numpy数组对象，二进制文件
        # npy春初多个Numpy数组对象，并且可以使用字典形式进行访问，二进制文件
        item_review_vec = np.load(os.path.join(data_path, 'item_review.npy'), allow_pickle=True).item()  # 提取项目评论特征
        vis_vec = np.load(os.path.join(data_path, 'image_feature.npy'), allow_pickle=True)  # 加载图片特征数据集
        filename = data_path + '/train.csv'  # 训练数据集 CSV 文件路径
        filename_test = data_path + '/test.csv'  # 测试数据集 CSV 文件路径
        df = pd.read_csv(filename, index_col=None, usecols=None)  # 读取训练数据集 CSV 文件
        df_test = pd.read_csv(filename_test, index_col=None, usecols=None)  # 读取测试数据集 CSV 文件
        num_items = 0  # 记录训练集和测试集的总的商品数量
        asin_i_dic = {}  # ASIN 和 itemID 的字典，key 是 itemID，value 是对应商品的 ASIN 编号

        # 遍历训练数据集并将其值分配给字典。
        for index, row in df.iterrows():
            asin, i = row['itemID'], int(row['encoded_itemID'])
            asin_i_dic[i] = asin
            num_items = max(num_items, i)

        # 遍历测试数据集并将其值分配给字典。
        for index, row in df_test.iterrows():
            asin, i = row['itemID'], int(row['encoded_itemID'])
            asin_i_dic[i] = asin
            num_items = max(num_items, i)

        item_review_features = []  # 商品的文本特征向量
        image_features = []  # 商品的图像特征向量

        # 遍历商品以获取其item特征。
        for i in range(num_items + 1):
            item_features = item_review_vec[asin_i_dic[i]]  # 获取 ASIN 编号为 i 的商品文本特征向量
            item_review_features.append(item_features)

            item_image_features = []  # 用于存储当前商品的所有图像特征
            for item_id, label_image_features in vis_vec:
                if item_id == asin_i_dic[i]:  # 仅当项目ID与当前商品匹配时，将图像特征添加到列表中
                    item_image_features.extend(label_image_features)  # 添加四种不同类别的图像特征到列表中

            image_features.append(item_image_features)  # 将当前商品的图像特征列表添加到总体图像特征中

        image_features = np.asarray(image_features, dtype=np.float32)
        print('Shape of image_features_np array:', image_features.shape)

        # 返回商品文本特征向量和图像特征向量
        return item_review_features, image_features


