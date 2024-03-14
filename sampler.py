import numpy as np
from scipy.sparse import lil_matrix
from torch.utils.data import Dataset, DataLoader


class WarpDataset(Dataset):
    """
    PyTorch Dataset类的子类，用于生成训练数据
    """
    def __init__(self, user_item_matrix, batch_size, n_negative, check_negative=True):
        self.user_item_matrix = user_item_matrix
        self.batch_size = batch_size
        self.n_negative = n_negative
        self.check_negative = check_negative
        self.user_item_pairs, self.user_to_positive_set = self._prepare_samples()

    def _prepare_samples(self):
        """
        准备正样本对和用户正样本集合字典
        """
        user_item_matrix = lil_matrix(self.user_item_matrix)
        user_item_pairs = np.asarray(user_item_matrix.nonzero()).T
        user_to_positive_set = {u: set(row) for u, row in enumerate(user_item_matrix.rows)}
        np.random.shuffle(user_item_pairs)  # 打乱user_item_pairs数组顺序
        return user_item_pairs, user_to_positive_set

    def __len__(self):
        return int(len(self.user_item_pairs) / self.batch_size)

    def __getitem__(self, index):
        user_positive_items_pairs = self.user_item_pairs[index * self.batch_size: (index + 1) * self.batch_size, :]
        # np.random.seed(10)  # 设置随机数种子，以确保每次运行生成的随机数序列相同
        negative_samples = np.random.randint(
            0,
            self.user_item_matrix.shape[1],
            size=(self.batch_size, self.n_negative))

        if self.check_negative:
            for user_positive, negatives, i in zip(user_positive_items_pairs,
                                                   negative_samples,
                                                   range(len(negative_samples))):
                user = user_positive[0]
                for j, neg in enumerate(negatives):
                    while neg in self.user_to_positive_set[user]:
                        negative_samples[i, j] = neg = np.random.randint(0, self.user_item_matrix.shape[1])

        return user_positive_items_pairs, negative_samples



