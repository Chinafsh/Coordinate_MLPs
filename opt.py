import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_path', type=str,default='./fox.jpg',
                        help='path to the image to reconstruct')
    parser.add_argument('--use_pe', default=False, action='store_true',
                        help='use positional encoding or not')
    
    parser.add_argument('--arch', type=str, default='identity',
                        choices=['relu', 'ff', 'siren', 'gaussian', 'quad', 'multi_quad', 'laplacian'],
                        help='network structure')
    
    
    parser.add_argument('--sc', type=float, default=10.,
                        help='gaussian embedding scale factor')
    parser.add_argument('--omega0', type=float, default=30.,
                        help='Siren omega0')
    parser.add_argument('--a', type=float, default=0.1,
                        help='gaussian activation a')
    parser.add_argument('--b', type=float, default=0.1,
                        help='quad activation b')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=2000,
                        help='number of epochs')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help ='experiment name')
    return parser.parse_args()