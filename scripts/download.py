
import argparse
import os
import subprocess

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idlist', required=True)
    parser.add_argument('--output', default='images')
    return parser.parse_args()

def load_idlist(path):
    with open(path, 'r') as fin:
        idlist = fin.read().strip().split('\n')
    return idlist

def save_rsync_filelist(idlist):
    _to_folder = lambda filename: str(int(os.path.splitext(filename)[0]) % 1000).zfill(4)
    filelist = [f'./original/{_to_folder(file)}/{file}' for file in idlist]
    with open('download-list.txt', 'w') as fout:
        fout.write('\n'.join(filelist))

def download(output):
    if not os.path.exists(output):
        os.makedirs(output)
    command = f'rsync -r --files-from=download-list.txt rsync://176.9.41.242:873/danbooru2021/ {output}'
    cp = subprocess.run(command, shell=True)
    while cp.returncode != 0:
        cp = subprocess.run(command, shell=True)

def main():
    args = get_args()

    idlist   = load_idlist(args.idlist)
    save_rsync_filelist(idlist)
    download(args.output)

if __name__=='__main__':
    main()
