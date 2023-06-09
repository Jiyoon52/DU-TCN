# -*- coding: utf-8 -*-
"""
@author: jiyoon
"""

from configs.configs import get_parser
import trainer
from itertools import product
import argparse
import warnings
from tqdm import tqdm
warnings.filterwarnings(action='ignore')


def main(args: argparse,
         threshold_list: list,
         type_list: list):

    tr = trainer.Train_Inference(args)
    tr.Train()

    for threshold, type_name in product(threshold_list, type_list):
        tr.Inference_train(threshold=threshold)
        tr.Inference_test(type=type_name)


def comb_product(comb):
    return (dict(zip(comb.keys(), values)) for values in product(*comb.values()))


if __name__ == '__main__':

    args = get_parser(method='tcn_sngp').parse_args()
    setting_dict = {'norm_multiplier': [0.4],  # 0.6
                    'gp_kernel_scale': [0.8],  # 0.8
                    'gp_cov_ridge_penalty': [6],  # 0.8
                    'num_inducing': [512],
                    'noise_ratio': ['0.2', '0.4', '0.6', '0.8'],
                    'scenario': ['T_IBN', 'TRN_IBN', 'VTRN_IBN'],
                    'gp_cov_momentum': [-1],
                    'nb_stacks': [3]
                    }

    threshold_list = [85, 90, 95, 99, 100]
    type_list = [1, 2]

    params = comb_product(setting_dict)
    for param in tqdm(params):

        args.__dict__.update(param)
        main(args, threshold_list, type_list)
