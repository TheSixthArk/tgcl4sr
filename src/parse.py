import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    ### 基本参数
    parser.add_argument('--data_name', type=str, default='Games')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tolerance', type=int, default=10)
    parser.add_argument('--mname', type=str, default='tgcl4sr')
    ### MainModel参数
    parser.add_argument('--cl_weight', type=float)
    parser.add_argument('--mmd_weight', type=float)
    parser.add_argument('--seq_head', type=int, default=2)
    parser.add_argument('--seq_layers', type=int, default=2)
    parser.add_argument('--seq_dropout', type=float, default=0.5)
    parser.add_argument('--temp', type=float)
    parser.add_argument('--sigma', type=float)
    ### TAT参数
    parser.add_argument('--sample_size', type=list, default=[20, 20])
    return parser.parse_args()