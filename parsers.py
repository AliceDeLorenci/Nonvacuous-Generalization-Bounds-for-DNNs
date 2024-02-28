import argparse


def get_main_parser():
    parser = argparse.ArgumentParser(description="Parser for main training loop")

    parser.add_argument('--n_layers', type=int, default=2, help='Number of hidden layers in the networks')
    parser.add_argument('--nhid', type=int, default=600, help='Number of hidden units in a hidden layer')
    parser.add_argument('--nin', type=int, default=784, help='Number of input dim')
    parser.add_argument('--nout', type=int, default=1, help='Number of outputs')
    
    parser.add_argument('--lr2', type=float, default=0.001, help='Learning rate 2')
    parser.add_argument('--sigma_init', type=float, default=0.1, help='Initial value of s') 
    parser.add_argument('--T', type=int, default=5000, help='Number of iterations of second loop')
    return parser.parse_args() 
