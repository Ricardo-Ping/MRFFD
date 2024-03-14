"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/4/27 9:23
@File : model.py
@function :
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from args import parse_args

args = parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionModel(nn.Module):
    def __init__(self, input_size):
        super(AttentionModel, self).__init__()
        self.user_linear = nn.Linear(4 * input_size, 3)
        self.item_linear = nn.Linear(2 * input_size, 1)
        self.textual_linear = nn.Linear(2 * input_size, 1)
        self.visual_linear = nn.Linear(2 * input_size, 1)

    def forward(self, user, item, textual, visual):
        user = user.to(device)
        item = item.to(device)
        textual = textual.to(device)
        visual = visual.to(device)

        user_input = torch.cat([user, item, textual, visual], dim=1)

        normalized_input = F.normalize(user_input, p=2, dim=1)

        user_output_h = torch.tanh(self.user_linear(normalized_input))

        item_input = torch.cat([user, item], dim=1)
        item_normalized_input = F.normalize(item_input, p=2, dim=1)
        item_output_h = F.relu(self.item_linear(item_normalized_input))

        textual_input = torch.cat([user, textual], dim=1)
        textual_normalized_input = F.normalize(textual_input, p=2, dim=1)
        textual_output_h = F.relu(self.textual_linear(textual_normalized_input))

        visual_input = torch.cat([user, visual], dim=1)
        visual_normalized_input = F.normalize(visual_input, p=2, dim=1)
        visual_output_h = F.relu(self.visual_linear(visual_normalized_input))

        item_scores = torch.exp(item_output_h.squeeze(1))
        textual_scores = torch.exp(textual_output_h.squeeze(1))
        visual_scores = torch.exp(visual_output_h.squeeze(1))

        user_scores = F.softmax(user_output_h, dim=1)

        # 归一化注意力分数
        item_scores = item_scores / torch.sum(item_scores)
        textual_scores = textual_scores / torch.sum(textual_scores)
        visual_scores = visual_scores / torch.sum(visual_scores)

        # 合并注意力分数
        merged_output = torch.stack([item_scores, textual_scores, visual_scores], dim=1)

        # 相加
        weighted_output = user_scores + merged_output

        # 计算注意力权重
        attention_weights = F.softmax(weighted_output, dim=1)

        return attention_weights


class TextAttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextAttentionModel, self).__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, text_features_list):
        project_embeddings_list = []

        for project_features in text_features_list:
            u_c = self.relu(self.linear(project_features))  # (n,1024)
            a_c_normalized = F.softmax(u_c, dim=1)  # 使用注意力线性层，并在评论维度上进行softmax操作,得到注意力分数
            weighted_features = torch.sum(a_c_normalized * project_features, dim=0)
            project_embeddings_list.append(weighted_features.unsqueeze(0))

        project_embeddings = torch.cat(project_embeddings_list, dim=0)

        return project_embeddings


class CategoryAttentionModel(nn.Module):
    def __init__(self, input_dim, num_categories, output_dim):
        super(CategoryAttentionModel, self).__init__()
        self.num_categories = num_categories
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim).to(device)  # 转换矩阵 W_beta
        self.relu = nn.ReLU().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

        self.fc = nn.Linear(output_dim, output_dim).to(device)  # 全连接层

        self.u_beta = nn.Parameter(torch.randn(num_categories, 1)).to(device)  # 上下文向量 u_beta，设为模型参数

    def forward(self, image_features):
        f_ic = image_features.view(-1, self.num_categories,
                                   self.output_dim)  # 将 image_features 转换为形状为 (batch_size, num_categories, output_dim)
        # print("f_ic shape:", f_ic.shape)

        ui_c = self.relu(self.linear(f_ic))  # 计算 ui_c
        # print("ui_c shape:", ui_c.shape)

        a_ic = torch.exp(ui_c * self.u_beta)  # 计算 a_ic
        # print("a_ic shape:", a_ic.shape)

        a_ic_sum = torch.sum(a_ic, dim=1, keepdim=True)  # 求和项的分母，对每个样本进行求和

        a_ic_normalized = a_ic / a_ic_sum  # 归一化得到注意力权重
        # print("a_ic_normalized shape:", a_ic_normalized.shape)

        f_i = torch.sum(a_ic_normalized * f_ic, dim=1)  # 按权重加权并求和得到 f_i
        # print("f_i shape:", f_i.shape)

        f_i = self.fc(f_i)  # 全连接层输出
        # print("output shape:", f_i.shape)

        return f_i


class DMRL(nn.Module):
    def __init__(self,
                 n_users,  # 用户数量
                 n_items,  # 项目数量
                 embed_dim=20,  # 嵌入维度
                 batch_size=10,  # 批处理大小
                 imagefeatures=None,  # 图像特征
                 textualfeatures=None,  # 文本特征
                 decay_r=1e-4,  # 正则项系数
                 decay_c=1e-3,  # 自适应学习率控制中的幂次
                 master_learning_rate=0.0001,  # 主学习率
                 hidden_layer_dim_a=256,  # 神经网络隐藏层维度A
                 hidden_layer_dim_b=256,  # 神经网络隐藏层维度B
                 dropout_rate_a=0.2,  # 神经网络隐藏层A的dropout率
                 dropout_rate_b=0.2,  # 神经网络隐藏层B的dropout率
                 ):
        super(DMRL, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.batch_size = batch_size

        self.master_learning_rate = master_learning_rate
        self.hidden_layer_dim_a = hidden_layer_dim_a
        self.hidden_layer_dim_b = hidden_layer_dim_b
        self.dropout_rate_a = dropout_rate_a
        self.dropout_rate_b = dropout_rate_b
        self.decay_r = decay_r
        self.decay_c = decay_c
        self.n_factors = args.n_factors  # 因子数量
        self.num_neg = args.num_neg  # 负采样数量

        if imagefeatures is not None:
            self.image_feature_model = CategoryAttentionModel(input_dim=1024, num_categories=4, output_dim=1024)
            with torch.no_grad():
                self.imagefeatures = torch.tensor(imagefeatures, dtype=torch.float32).to(device)
            self.image_features = self.image_feature_model(self.imagefeatures.to(device))
            # 保存特征为.npy文件
            # image_features_numpy = self.image_features.detach().cpu().numpy()
            # np.save("attention_image_features.npy", image_features_numpy)
            # print("保存成功")
            self.feature_projection_visual = nn.Sequential(
                nn.Linear(self.image_features.shape[1], 2 * self.hidden_layer_dim_a),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout_rate_a),
                nn.Linear(2 * self.hidden_layer_dim_a, self.embed_dim)
            ).to(device)
            self.image_feature_model.to(device)  # 将CategoryAttentionModel移至设备上
            self.parameters = list(self.parameters()) + list(self.image_feature_model.parameters())  # 添加参数到模型的参数列表
        else:
            self.feature_projection_visual = None

        if textualfeatures is not None:  # 是否存在文本特征
            self.text_feature_model = TextAttentionModel(input_dim=1024, output_dim=1024).to(device)
            with torch.no_grad():
                self.text_features_list = [torch.tensor(features, dtype=torch.float32).to(device) for features in
                                           textualfeatures]
            self.textualfeatures = self.text_feature_model(self.text_features_list)  # (batch_size, 128)
            # print('textualfeatures:', self.textualfeatures.shape)
            self.feature_projection_textual = nn.Sequential(
                nn.Linear(self.textualfeatures.shape[1], 2 * self.hidden_layer_dim_b),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout_rate_b),
                nn.Linear(2 * self.hidden_layer_dim_b, self.embed_dim)
            ).to(device)
        else:
            self.feature_projection_textual = None

        self.user_embeddings = nn.Embedding(self.n_users, self.embed_dim).to(device)
        self.item_embeddings = nn.Embedding(self.n_items, self.embed_dim).to(device)
        # using Xavier initialization
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

        attention_dim = int(embed_dim / args.n_factors)
        # print(attention_dim)
        self.attention_model = AttentionModel(attention_dim).to(device)

    def forward(self, user_positive_items_pairs, negative_samples):
        # print('textualfeatures:', self.textualfeatures.shape)
        users = self.user_embeddings(user_positive_items_pairs[:, 0].long()).to(device)
        pos_items = self.item_embeddings(user_positive_items_pairs[:, 1].long()).to(device)
        # self.textualfeatures = self.textualfeatures.clone().detach().to(device)
        # self.imagefeatures = self.imagefeatures.clone().detach().to(device)

        pos_i_f = self.textualfeatures[user_positive_items_pairs[:, 1].long().to(device)]
        pos_i_f = self.feature_projection_textual(pos_i_f).to(device)
        pos_i_v = self.image_features[user_positive_items_pairs[:, 1].long().to(device)]
        pos_i_v = self.feature_projection_visual(pos_i_v).to(device)
        neg_items = self.item_embeddings(negative_samples.view(-1).long()).view(-1, self.embed_dim).to(device)
        neg_i_f = self.feature_projection_textual(
            self.textualfeatures[negative_samples.view(-1).long().to(device)]).view(-1, self.embed_dim).to(device)

        neg_i_v = self.feature_projection_visual(self.image_features[negative_samples.view(-1).long().to(device)]).view(
            -1, self.embed_dim).to(device)

        items = torch.cat([pos_items, neg_items], 0).to(device)
        textual_f = torch.cat([pos_i_f, neg_i_f], 0).to(device)
        visual_f = torch.cat([pos_i_v, neg_i_v], 0).to(device)

        user_a = users.repeat(self.num_neg + 1, 1).to(device)
        user_factor_embedding = torch.chunk(users, self.n_factors, dim=1)
        item_factor_embedding = torch.chunk(items, self.n_factors, dim=1)
        item_factor_embedding_p = torch.chunk(pos_items, self.n_factors, dim=1)

        textual_factor_embedding = torch.chunk(textual_f, self.n_factors, dim=1)
        textual_factor_embedding_p = torch.chunk(pos_i_f, self.n_factors, dim=1)
        visual_factor_embedding = torch.chunk(visual_f, self.n_factors, dim=1)
        visual_factor_embedding_p = torch.chunk(pos_i_v, self.n_factors, dim=1)

        cor_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        for i in range(0, self.n_factors - 1):
            x = visual_factor_embedding_p[i].to(device)
            y = visual_factor_embedding_p[i + 1].to(device)
            cor_loss += self._create_distance_correlation(x, y)

            x = textual_factor_embedding_p[i].to(device)
            y = textual_factor_embedding_p[i + 1].to(device)
            cor_loss += self._create_distance_correlation(x, y)

            x = user_factor_embedding[i].to(device)
            y = user_factor_embedding[i + 1].to(device)
            cor_loss += self._create_distance_correlation(x, y)

            x = item_factor_embedding_p[i].to(device)
            y = item_factor_embedding_p[i + 1].to(device)
            cor_loss += self._create_distance_correlation(x, y)

        # 每个样本在所有因子上的平均距离相关性
        cor_loss /= ((self.n_factors + 1.0) * self.n_factors / 2)

        p_item, n_item = torch.split(items, [self.batch_size, self.num_neg * self.batch_size], 0)

        user_ap, user_an = torch.split(user_a, [self.batch_size, self.num_neg * self.batch_size], 0)

        user_factor_embedding_a = torch.chunk(user_a, self.n_factors, dim=1)
        user_factor_embedding_ap = torch.chunk(user_ap, self.n_factors, dim=1)
        user_factor_embedding_an = torch.chunk(user_an, self.n_factors, dim=1)

        p_item_factor_embedding = torch.chunk(p_item, self.n_factors, dim=1)
        n_item_factor_embedding = torch.chunk(n_item, self.n_factors, dim=1)

        pos_scores, neg_scores = [], []

        for i in range(0, self.n_factors):
            weights = self.attention_model(user_factor_embedding_a[i].to(device), item_factor_embedding[i].to(device),
                                           textual_factor_embedding[i].to(device),
                                           visual_factor_embedding[i].to(device)).to(device)

            p_weights, n_weights = torch.split(weights, [self.batch_size, self.num_neg * self.batch_size], 0)

            textual_trans = textual_factor_embedding[i].to(device)

            p_textual_trans, n_textual_trans = torch.split(textual_trans,
                                                           [self.batch_size, self.num_neg * self.batch_size], 0)

            visual_trans = visual_factor_embedding[i].to(device)

            p_visual_trans, n_visual_trans = torch.split(visual_trans,
                                                         [self.batch_size, self.num_neg * self.batch_size], 0)

            p_score = (p_weights[:, 1] * torch.nn.functional.softplus(
                torch.sum(user_factor_embedding_ap[i] * p_textual_trans.to(device), dim=1)) +
                       p_weights[:, 2] * torch.nn.functional.softplus(
                        torch.sum(user_factor_embedding_ap[i] * p_visual_trans.to(device), dim=1)) +
                       p_weights[:, 0] * torch.nn.functional.softplus(
                        torch.sum(user_factor_embedding_ap[i] * p_item_factor_embedding[i].to(device), dim=1))).to(
                device)

            pos_scores.append(torch.unsqueeze(p_score, dim=1))

            n_score = (n_weights[:, 1] * torch.nn.functional.softplus(
                torch.sum(user_factor_embedding_an[i] * n_textual_trans.to(device), dim=1)) +
                       n_weights[:, 2] * torch.nn.functional.softplus(
                        torch.sum(user_factor_embedding_an[i] * n_visual_trans.to(device), dim=1)) +
                       n_weights[:, 0] * torch.nn.functional.softplus(
                        torch.sum(user_factor_embedding_an[i] * n_item_factor_embedding[i].to(device), dim=1))).to(
                device)

            neg_scores.append(torch.unsqueeze(n_score, dim=1))

        pos_s = torch.cat(pos_scores, dim=1)

        neg_s = torch.cat(neg_scores, dim=1)

        # 这里是l2正则化
        regularizer = torch.tensor(0, dtype=torch.float32, device=device)
        regularizer += torch.nn.functional.mse_loss(users, torch.zeros_like(users), reduction='sum')
        regularizer += torch.nn.functional.mse_loss(pos_items, torch.zeros_like(pos_items), reduction='sum')
        regularizer += torch.nn.functional.mse_loss(neg_items, torch.zeros_like(neg_items), reduction='sum')
        regularizer += torch.nn.functional.mse_loss(pos_i_v, torch.zeros_like(pos_i_v), reduction='sum')
        regularizer += torch.nn.functional.mse_loss(neg_i_v, torch.zeros_like(neg_i_v), reduction='sum')
        regularizer += torch.nn.functional.mse_loss(pos_i_f, torch.zeros_like(pos_i_f), reduction='sum')
        regularizer += torch.nn.functional.mse_loss(neg_i_f, torch.zeros_like(neg_i_f), reduction='sum')
        regularizer = regularizer / self.batch_size

        pos_score = torch.sum(pos_s, dim=1)

        negtive_score = torch.max(torch.reshape(torch.sum(neg_s, dim=1), [self.batch_size, self.num_neg]), dim=1)[0]

        loss_per_pair = torch.nn.functional.softplus(-(pos_score - negtive_score))

        loss = torch.sum(loss_per_pair)

        all_loss = loss + self.decay_r * regularizer + self.decay_c * cor_loss

        return all_loss

    def predict(self, score_user_ids):
        users = self.user_embeddings(score_user_ids.to(device)).unsqueeze(1)
        item = self.item_embeddings.weight.to(device).unsqueeze(0)
        textual = self.feature_projection_textual(self.textualfeatures.to(device)).unsqueeze(0)
        visual = self.feature_projection_visual(self.image_features.to(device)).unsqueeze(0)
        # visual_clone = visual.squeeze(0)
        # print(visual_clone.shape)
        # # 保存特征为.npy文件
        # visual_numpy = visual_clone.detach().cpu().numpy()
        # np.save("visual_numpy.npy", visual_numpy)
        # print("保存成功")

        item_expand = item.repeat(users.size(0), 1, 1).reshape(-1, self.embed_dim)
        textual_expand = textual.repeat(users.size(0), 1, 1).reshape(-1, self.embed_dim)
        visual_expand = visual.repeat(users.size(0), 1, 1).reshape(-1, self.embed_dim)
        users_expand = users.repeat(1, item.size(1), 1).reshape(-1, self.embed_dim)

        user_expand_factor_embedding = torch.chunk(users_expand, self.n_factors, 1)
        item_expand_factor_embedding = torch.chunk(item_expand, self.n_factors, 1)
        textual_expand_factor_embedding = torch.chunk(textual_expand, self.n_factors, 1)
        visual_expand_factor_embedding = torch.chunk(visual_expand, self.n_factors, 1)

        factor_scores = []
        factor_sc = []
        factor_ws = []

        for i in range(0, self.n_factors):
            weights = self.attention_model(user_expand_factor_embedding[i], item_expand_factor_embedding[i],
                                           textual_expand_factor_embedding[i], visual_expand_factor_embedding[i])
            textual_trans = textual_expand_factor_embedding[i]
            visual_trans = visual_expand_factor_embedding[i]

            f_score = (weights[:, 1] * torch.nn.functional.softplus(
                torch.sum(user_expand_factor_embedding[i] * textual_trans, dim=1)) +
                       weights[:, 2] * torch.nn.functional.softplus(
                        torch.sum(user_expand_factor_embedding[i] * visual_trans, dim=1)) +
                       weights[:, 0] * torch.nn.functional.softplus(
                        torch.sum(user_expand_factor_embedding[i] * item_expand_factor_embedding[i], dim=1)))

            factor_scores.append(torch.unsqueeze(f_score, 1))

            factor_sc.append([weights[:, 0] * torch.nn.functional.softplus(
                torch.sum(user_expand_factor_embedding[i] * item_expand_factor_embedding[i], dim=1)),
                              weights[:, 1] * torch.nn.functional.softplus(
                                  torch.sum(user_expand_factor_embedding[i] * textual_trans, dim=1)),
                              weights[:, 2] * torch.nn.functional.softplus(
                                  torch.sum(user_expand_factor_embedding[i] * visual_trans, dim=1))])

            factor_ws.append(weights)

        factor_s = torch.cat(factor_scores, 1)
        scores = torch.reshape(torch.sum(factor_s, dim=1), (users.shape[0], -1))

        return scores, factor_sc, factor_ws

    # 用于计算损失的权重向量
    # def _create_weight(self, user, item, textual, visual):
    #     # print("user:", user.shape)
    #     # print("item:", item.shape)
    #     # print("textual:", textual.shape)
    #     # print("visual:", visual.shape)
    #     user = user.to(device)
    #     item = item.to(device)
    #     textual = textual.to(device)
    #     visual = visual.to(device)
    #
    #     # input是把四个张量按照特征维度连接起来，表示用户-物品交互数据，
    #     # 使用L2范数来缩放张量中的每一个特征
    #     input = torch.nn.functional.normalize(torch.cat([user, item, textual, visual], 1), p=2, dim=1).to(device)
    #
    #     # 第一个全连接层将输入映射到一个低维空间，
    #     # 第二个全连接层通过对第一个全连接层的输出进行线性组合来计算最终输出
    #     output_h = torch.nn.functional.tanh(torch.nn.Linear(input.shape[1], 3).to(device)(input))
    #     output = torch.nn.Linear(3, 3, bias=False).to(device)(output_h)
    #
    #     probability = torch.nn.functional.softmax(output, dim=1).to(device)
    #     # print("probability:", probability.shape)
    #
    #     return probability

    # 计算两组嵌入向量之间的距离相关性
    def _create_distance_correlation(self, X1, X2):
        def _create_centered_distance(X):
            r = torch.sum(torch.square(X), 1, keepdim=True)
            D = torch.sqrt(torch.maximum(r - 2 * torch.matmul(X, X.transpose(1, 0)) + r.transpose(1, 0),
                                         torch.tensor([0.0]).to(device)) + 1e-8)
            D = D - torch.mean(D, dim=0, keepdim=True) - torch.mean(D, dim=1, keepdim=True) + torch.mean(D)
            return D

        def _create_distance_covariance(D1, D2):
            n_samples = torch.tensor(D1.shape[0], dtype=torch.float32).to(device)
            dcov = torch.sqrt(
                torch.maximum(torch.sum(D1 * D2) / (n_samples * n_samples), torch.tensor([0.0]).to(device)) + 1e-8)
            return dcov

        X1 = X1.to(device)
        X2 = X2.to(device)
        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        dcor = dcov_12 / (torch.sqrt(torch.max(dcov_11 * dcov_22, torch.tensor([0.0]).to(device))) + 1e-10)
        dcor = dcor.squeeze()  # 将形状为 [1] 的张量转换为标量
        # print("dcor:", dcor.shape)
        return dcor

    # 使用函数“torch.clamp”将每个向量的范数限制在1.0以内，同时保持其方向不变
    def clip_by_norm_op(self):
        clip_grad_norm_(self.user_embeddings.weight.to(device), 1.0)
        clip_grad_norm_(self.item_embeddings.weight.to(device), 1.0)
