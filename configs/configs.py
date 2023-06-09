import sys, os
sys.path.append(os.getcwd())

import argparse
from configs.tcn_sngp import tcn_sngp_parser
from configs.tcn import tcn_parser
from configs.hsn import hsn_parser
from configs.osit import osit_parser


def get_parser(method: str):
    
    if method == 'tcn_sngp':
        parents = [tcn_sngp_parser()]
        
        
    elif method == 'tcn':
         parents = [tcn_parser()]

    elif method == 'hsn':
        parents = [hsn_parser()]
        
    elif method == 'osit':
        parents = [osit_parser()]
        
    # elif method == 'something':
    #     parents = [something_parser()]
        

    else:
        raise NotImplementedError(f'there is no {method} method.')
    
    parser = argparse.ArgumentParser(description="Open Set Recognition",
                                     parents=parents)
    # Experiments setting
    parser.add_argument('--method', type=str, default='tcn_sngp',
                        choices=['tcn_sngp', 'tcn', 'hsn'])
    parser.add_argument('--seed', type=int, default=2022, help='random seed (default: 2022)')
    
    # Directory hyperparameter
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--save-dir', type=str, default='./save/')
    parser.add_argument('--result-dir', type=str, default='./results/')
    parser.add_argument('--ckpt-dir', type=str, default='./ckpt/')
    
    # Dataset hyperparameter
    parser.add_argument('--num-known', type=int, default=3, help='set under 7 classes') # num of unknown = 1
    parser.add_argument('--unknown-cls-idx', type=int, default=3, help='set under 7 classes') # num of unknown = 1
    parser.add_argument('--scenario', type=str, default='T_IBN', choices=['T_IBN', 'TRN_IBN', 'VTRN_IBN'])
    parser.add_argument('--noise-ratio', type=str, default='0.2', choices=['0.02', '0.2', '0.3,', '0.4', 
                                                                           '0.5', '0.6', '0.7', '0.8'])
    
    return parser