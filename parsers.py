import argparse


def get_main_parser():
    parser = argparse.ArgumentParser(description="Parser for main training loop")

    parser.add_argument('--nn_type', type=str, default='mlp', choices=['mlp', 'cnn'], help='Type of neural network to use')
    parser.add_argument('--nout', type=int, default=1, choices=[1, 10], help='Number of outputs')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of hidden layers in the networks')

    # CNN specifications
    parser.add_argument('--nin_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--kernel_size', type=int, default=3, help='Size of the convolutional kernel')
    parser.add_argument('--nfilters', type=int, default=16, help='Number of filters in the convolutional layers')

    # MLP specifications
    parser.add_argument('--nin', type=int, default=784, help='Input dimension')
    parser.add_argument('--nhid', type=int, default=600, help='Number of hidden units in a hidden layer')
    
    # Initial training specifications
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for SGD optimization')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for SGD optimization')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for SGD optimization')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay for SGD optimization')
    
    # PAC-Bayes bound optimization specifications
    parser.add_argument('--lr2', type=float, default=0.001, help='Learning rate for PAC-Bayes bound optimization')
    parser.add_argument('--sigma_init', type=float, default=1., help='Scaling to apply to the initial value of s')  # 1 for true-labels, 0.1 for random-labels
    parser.add_argument('--T', type=int, default=5000, help='Number of iterations for PAC-Bayes bound optimization') # paper uses 200 000
    parser.add_argument('--nb_snns', type=int, default=200, help='Number of SNNs to sample for MC approximation') # paper uses 150 000
    parser.add_argument('--scheduler_patience', type=int, default=50, help='Patience of scheduler')
    
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    return parser.parse_args() 
