
import argparse
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idlist', required=True)
    parser.add_argument('--num-select', required=True, type=int)
    parser.add_argument('--output', default='./random_selected.txt')
    parser.add_argument('--seed', type=int)
    return parser.parse_args()

def load_idlist(path):
    with open(path, 'r') as fin:
        idlist = fin.read().strip().split('\n')
    return idlist

def main():
    args = get_args()

    idlist = load_idlist(args.idlist)

    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(idlist)
    selected = idlist[:args.num_select]

    with open(args.output, 'w') as fout:
        fout.write('\n'.join(selected))

if __name__=='__main__':
    main()
