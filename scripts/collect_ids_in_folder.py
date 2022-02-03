
import os, glob
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--output', default='ids.txt')
    return parser.parse_args()

def main():
    args = get_args()

    files = glob.glob(os.path.join(args.folder, '**', '*'), recursive=True)
    ids = [os.path.splitext(os.path.basename(file))[0] for file in files if os.path.isfile(file)]

    with open(args.output, 'w') as fout:
        fout.write('\n'.join(ids))

if __name__=='__main__':
    main()
