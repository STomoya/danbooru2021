
from __future__ import annotations

import argparse
import glob
import os

from joblib import delayed, Parallel
from tqdm import tqdm

from src import meta_utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./danboorumeta')
    parser.add_argument('--tags', required=True, nargs='+')
    parser.add_argument('--exclude-tags', nargs='*')
    parser.add_argument('--output', default='./collected_by_tag.txt')
    return parser.parse_args()

def filter_by_tag(metadata: dict, target: set, exclude: set|None):
    if 'id' in metadata.keys():
        tags = set(metadata['tag_string'].strip().split(' '))
        if exclude is not None and exclude.issubset(tags):
            return None
        if target.issubset(tags):
            return f'{metadata["id"]}.{metadata["file_ext"]}'
    return None

def main():
    args = get_args()
    metafiles = glob.glob(os.path.join(args.root, '*.json'))
    target = set(args.tags)
    exclude = set(args.exclude_tags) if isinstance(args.exclude_tags, list) else None
    id_list = []
    for metafile in tqdm(metafiles):
        metadata_dicts = meta_utils.load_as_dict(metafile)
        satisfied = Parallel(n_jobs=-1)(
            delayed(filter_by_tag)(metadata, target, exclude) for metadata in metadata_dicts)
        id_list.extend(list(set(satisfied) - {None}))

    with open(args.output, 'w') as fout:
        fout.write('\n'.join(id_list))

if __name__=='__main__':
    main()
