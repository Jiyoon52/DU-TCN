import argparse

def tcn_sngp_parser():
    parser = argparse.ArgumentParser(description='TCN_SNGP configs', add_help=False)
    
    # Training hyperparameter
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--kernel_size', type=int, default=2,
                        help='The size of the convolutional kernel')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1e-3,
                        help='Float between 0 and 1. Fraction of the input units to drop.')
    parser.add_argument('--nb_stacks', type=int, default=5,
                        help='The number of stacks of residual blocks to use')
    parser.add_argument('--norm_multiplier', type=float, default=0.6)
    parser.add_argument('--gp_kernel_scale', type=float, default=0.8)
    parser.add_argument('--gp_cov_ridge_penalty', type=float, default=0.8)
    parser.add_argument('--nb_filters', type=int, default=32,
                        help='The number of filters to use in the convolutional layers')   
    parser.add_argument('--num_inducing', type=int, default=512)
    parser.add_argument('--gp_cov_momentum', type=float, default=-1)
    parser.add_argument('--normalize_input', type=bool, default=False)
    parser.add_argument('--scale_random_features', type=bool, default=False)
                        
    return parser