
import argparse
import os
from src import meta_utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', default='./danboorumeta/filelist.txt')
    parser.add_argument('--idlist', required=True)
    parser.add_argument('--output', default='./filtered_list.txt')
    parser.add_argument('--resized', default=False, action='store_true')
    return parser.parse_args()

def load_id_list(path):
    with open(path, 'r') as fin:
        id_list = fin.read().strip().split('\n')
    return id_list


def main():
    args = get_args()
    filelist = meta_utils.load_filelist(args.filelist)
    if not args.resized:
        filelist = [file for file in filelist if './original/' in file]
    else:
        filelist = [file for file in filelist if './512px/' in file]
    exist_idlist = set([os.path.basename(file) for file in filelist])
    idlist = set(load_id_list(args.idlist))

    existance_filtered_idlist = list(idlist & exist_idlist)

    with open(args.output, 'w') as fout:
        fout.write('\n'.join(existance_filtered_idlist))

if __name__=='__main__':
    main()
