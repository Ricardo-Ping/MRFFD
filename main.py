"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/4/26 22:53
@File : main.py
@function :
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from Dataset import Dataset
from args import parse_args
from model import DMRL
from sampler import WarpDataset
from evaluator import RecallEvaluator
from tqdm import tqdm
import logging
import time
import toolz
import datetime
import matplotlib.pyplot as plt
import random
from torch.backends import cudnn

# Set the random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = False  # if benchmark=True, deterministic will be False
cudnn.deterministic = True

# Get current timestamp for log filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"train_{timestamp}.log"

# Set up logging
logging.basicConfig(filename=log_filename,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

epoch_eval = 20  # 多少个epoch评估一次


def load_data():
    Filename = args.dataset
    Filepath = 'Data/' + Filename
    dataset = Dataset(Filepath)
    train_data, test_data = dataset.trainMatrix, dataset.testRatings
    textualfeatures, imagefeatures = dataset.textualfeatures, dataset.imagefeatures
    n_users, n_items = max(train_data.shape[0], test_data.shape[0]), max(train_data.shape[1], test_data.shape[1])
    train_num = dataset.train_num

    warp_dataset = WarpDataset(train_data, batch_size=args.batch_size, n_negative=args.num_neg, check_negative=True)
    train_loader = DataLoader(warp_dataset, batch_size=None, shuffle=True, num_workers=0)

    return train_loader, train_data, test_data, textualfeatures, imagefeatures, n_users, n_items, train_num


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is triggered at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def train_and_evaluate(model, train_loader, train_data, train_num, test_data):
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam([
    #     {'params': model.user_embeddings.parameters()},
    #     {'params': model.item_embeddings.parameters()},
    #     {'params': model.text_feature_model.parameters()}
    # ], lr=args.learning_rate)
    optimizer = torch.optim.Adam([
        {'params': model.user_embeddings.parameters()},
        {'params': model.item_embeddings.parameters()},
        {'params': model.image_feature_model.parameters()}

    ], lr=args.learning_rate)

    EVALUATION_EVERY_N_BATCHES = train_num // args.batch_size + 1
    best_value, stopping_step, should_stop = 0, 0, False
    test_users = np.asarray(list(set(test_data.nonzero()[0])), dtype=np.int32)
    rec_loger, pre_loger, ndcg_loger, hit_loger = [], [], [], []
    loss_log = []

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        model.train()
        t1 = time.time()
        running_loss = 0.0
        for i, (user_positive_items_pairs, negative_samples) in enumerate(train_loader):
            user_positive_items_pairs = user_positive_items_pairs.to(torch.float32).to(device)
            negative_samples = negative_samples.to(torch.float32).to(device)

            optimizer.zero_grad()
            # forward
            loss = model(user_positive_items_pairs, negative_samples)
            loss.backward(retain_graph=True)
            # loss.backward()
            model.clip_by_norm_op()
            optimizer.step()
            running_loss += loss.item()

        running_loss = running_loss / EVALUATION_EVERY_N_BATCHES
        loss_log.append(running_loss)  # 将损失值添加到 loss_log 列表中
        t2 = time.time()
        logger.info("Epoch {}, Loss: {:.5f}".format(epoch + 1, running_loss))

        # Evaluate
        model.eval()
        with torch.no_grad():
            if (epoch + 1) % epoch_eval == 0:
                with tqdm(total=len(test_users), desc="Evaluating", ncols=100) as pbar:

                    testresult = RecallEvaluator(model, train_data, test_data)
                    test_recalls, test_ndcg, test_hr, test_pr = [], [], [], []

                    for user_chunk in tqdm(toolz.partition_all(20, test_users), desc="User Chunks", ncols=100):
                        recalls, ndcgs, hit_ratios, precisions = testresult.eval(user_chunk)
                        test_recalls.extend(recalls)
                        test_ndcg.extend(ndcgs)
                        test_hr.extend(hit_ratios)
                        test_pr.extend(precisions)
                        pbar.update(len(user_chunk))

                    recalls = np.mean(test_recalls)
                    precisions = np.mean(test_pr)
                    hit_ratios = np.mean(test_hr)
                    ndcgs = np.mean(test_ndcg)

                    rec_loger.append(recalls)
                    pre_loger.append(precisions)
                    ndcg_loger.append(ndcgs)
                    hit_loger.append(hit_ratios)

                t3 = time.time()
                logger.info(
                    "Epoch {}: [{}s + {}s] Loss: {:.5f}, Recall: {:.5f}, Precision: {:.5f}, Hit Ratio: {:.5f}, NDCG: {:.5f}".format(
                        epoch + 1, round(t2 - t1, 1), round(t3 - t2, 1),
                        running_loss / (10 * EVALUATION_EVERY_N_BATCHES),
                        recalls,
                        precisions, hit_ratios, ndcgs))

                best_value, stopping_step, should_stop = early_stopping(recalls, best_value, stopping_step,
                                                                        expected_order='acc',
                                                                        flag_step=5)

                if should_stop:
                    break

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    best_rec_0 = max(recs)
    idx = list(recs).index(best_rec_0)
    logger.info("Best Iter = recall:%.5f, pres:%.5f, hr:%.5f, ndcg:%.5f" % (recs[idx], pres[idx], hit[idx], ndcgs[idx]))
    logger.handlers.clear()

    # Plot loss
    plt.figure()
    plt.plot(range(len(loss_log)), [loss / (10 * EVALUATION_EVERY_N_BATCHES) for loss in loss_log])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss_plot.png')
    plt.show()

    # Plot evaluation metrics
    plt.figure()
    plt.plot(range(0, len(rec_loger) * epoch_eval, epoch_eval), rec_loger, label='Recall')
    plt.plot(range(0, len(pre_loger) * epoch_eval, epoch_eval), pre_loger, label='Precision')
    plt.plot(range(0, len(ndcg_loger) * epoch_eval, epoch_eval), ndcg_loger, label='NDCG')
    plt.plot(range(0, len(hit_loger) * epoch_eval, epoch_eval), hit_loger, label='Hit Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Evaluation Metrics')
    plt.legend()
    plt.savefig('evaluation_metrics_plot.png')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    train_loader, train_data, test_data, textualfeatures, imagefeatures, n_users, n_items, train_num = load_data()

    # 创建 PyTorch 的 GPU 设备对象
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DMRL(n_users,
                 n_items,
                 embed_dim=128,
                 batch_size=args.batch_size,
                 imagefeatures=imagefeatures,
                 textualfeatures=textualfeatures,
                 decay_r=args.decay_r,
                 decay_c=args.decay_c,
                 master_learning_rate=args.learning_rate,
                 hidden_layer_dim_a=args.hidden_layer_dim_a,
                 hidden_layer_dim_b=args.hidden_layer_dim_b,
                 dropout_rate_a=args.dropout_a,
                 dropout_rate_b=args.dropout_b).to(device)

    train_and_evaluate(model, train_loader, train_data, train_num, test_data)
