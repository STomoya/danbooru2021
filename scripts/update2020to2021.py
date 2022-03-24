
import argparse
import os, glob
import shutil
import subprocess

import cv2
from joblib import Parallel, delayed

TEMP = './temp'
TOTAL_FOLDERS = 1000
PARTIAL_FILENAME = './partial-filelist.txt'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', default=512, type=int, help='Length of the long side of the image after resizing in pixels.')
    parser.add_argument('--save-as', default='.jpg', help='Extension of the save image.')
    return parser.parse_args()

def prepair_folders(output_root):
    for i in range(TOTAL_FOLDERS):
        folder = os.path.join(output_root, str(i).zfill(4))
        if not os.path.exists(folder):
            os.mkdir(folder)

def start_from():
    if os.path.exists('done'):
        with open('done', 'r') as fin:
            start = int(fin.read()) + 1
    else: start = 0
    return start

def update_done(current):
    with open('done', 'w') as fout:
        fout.write(str(current))

def make_partial_filelist(current:int, filelist):
    folder = str(current).zfill(4)
    filelist = [file for file in filelist if f'./original/{folder}' in file]
    with open(PARTIAL_FILENAME, 'w') as fout:
        fout.write('\n'.join(filelist))

'''online'''

def download(output=TEMP):
    command = f'rsync -r --files-from={PARTIAL_FILENAME} rsync://176.9.41.242:873/danbooru2021/ {output}'
    cp = subprocess.run(command, shell=True)
    while cp.returncode != 0:
        cp = subprocess.run(command, shell=True)


'''offline'''

def resize(image, size):
    height, width = image.shape[:2]
    if height >= width:
        scale = size / height
    else:
        scale = size / width

    image = cv2.resize(image, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_LANCZOS4)
    return image

# calc folder from filename (id)
# If file names are {10001, 19999} then the folders will be {0001, 0999}
filename2folder = lambda filename: str(int(filename) % 1000).zfill(4)

def try_load_resize_save(src, output, size=512, saved_as='.jpg'):
    filename: str = os.path.splitext(os.path.basename(src))[0]
    if not filename.isdigit(): # filenames should be <id: integer number>.<extension>
        print(f'Invalid filename {filename}')
        return

    # try to load image. If not possible, abort.
    try:
        image = cv2.imread(src, cv2.IMREAD_COLOR)
    except Exception as e:
        print(e)
        return
    if image is None:
        print(f'None when reading {filename}')
        return

    image = resize(image, size)
    dst = os.path.join(output, filename2folder(filename), filename + saved_as)
    cv2.imwrite(dst, image)

def main():
    args = get_args()

    start = start_from()
    if start == 0:
        prepair_folders(args.output)
    if os.path.exists(TEMP):
        shutil.rmtree(TEMP)

    for intfolder in range(start, TOTAL_FOLDERS):
        make_partial_filelist(intfolder)
        download(TEMP)
        downloaded_files = glob.glob(os.path.join(TEMP, '**/*'), recursive=True)
        downloaded_files = [file for file in downloaded_files if os.path.isfile(file)]
        Parallel(n_jobs=-1)(delayed(try_load_resize_save)(file, args.output, args.size, args.save_as) for file in downloaded_files)
        update_done(intfolder)
        shutil.rmtree(TEMP)

if __name__=='__main__':
    main()
