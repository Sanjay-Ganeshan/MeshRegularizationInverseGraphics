import argparse
import os
import numpy as np
import torch
from easydict import EasyDict as ED


class Options:
    def __init__(self):
        self.args = None
        self.parse_args_det()

    def parse_args_det(self):
        self.args = ED()
        self.args.torch_seed = 5
        self.args.res_blocks = 3
        self.args.leaky_relu = 0.01
        self.args.convs = [16,32,64,64,128]
        self.args.pools = [0.0, 0.0, 0.0, 0.0]
        self.args.transfer_data = False
        self.args.overlap = 0
        self.args.init_weights = 0.002

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Point2Mesh options')
        # HYPER PARAMETERS - RECONSTRUCTION
        parser.add_argument('--torch-seed', type=int, metavar='N', default=5, help='torch random seed')
        # HYPER PARAMETERS - NETWORK
        parser.add_argument('--lr', type=float, metavar='1eN', default=1.1e-4, help='learning rate')
        parser.add_argument('--res-blocks', type=int, metavar='N', default=3, help='')
        parser.add_argument('--leaky-relu', type=float, metavar='1eN', default=0.01, help='slope for leaky relu')
        parser.add_argument('--convs', nargs='+', default=[16, 32, 64, 64, 128], type=int, help='convs to do')
        parser.add_argument('--pools', nargs='+', default=[0.0, 0.0, 0.0, 0.0], type=float,
                            help='percent to pool from original resolution in each layer')
        parser.add_argument('--transfer-data', action='store_true', help='')
        parser.add_argument('--overlap', type=int, default=0, help='overlap for bfs')
        parser.add_argument('--init-weights', type=float, default=0.002, help='initialize NN with this size')
        #
        self.args = parser.parse_args()

    def get_num_parts(self, num_faces):
        return 1

    def dtype(self):
        return torch.float32
