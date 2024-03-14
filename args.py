"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/4/27 9:34
@File : args.py
@function :
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run DMRL.')
    parser.add_argument('--dataset', nargs='?', default='Yelp', help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=1000, help='total_epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate.')
    parser.add_argument('--decay_r', type=float, default=1e-2, help='decay_r.')
    parser.add_argument('--decay_c', type=float, default=1e-0, help='decay_c.')
    parser.add_argument('--decay_p', type=float, default=0, help='decay_p.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--n_factors', type=int, default=4, help='Number of factors.')
    parser.add_argument('--num_neg', type=int, default=5, help='negative items')
    parser.add_argument('--hidden_layer_dim_a', type=int, default=256, help='Hidden layer dim a.')
    parser.add_argument('--hidden_layer_dim_b', type=int, default=128, help='Hidden layer dim b.')
    parser.add_argument('--dropout_a', type=float, default=0.2, help='dropout_a.')
    parser.add_argument('--dropout_b', type=float, default=0.2, help='dropout_b.')
    parser.add_argument('--num_threads', type=int, default=8, help='Number of threads to use for evaluation.')
    args = parser.parse_args()

    return args
