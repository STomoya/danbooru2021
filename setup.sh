#!/bin/bash

# download metadata
mkdir danboorumeta
rsync rsync://176.9.41.242:873/danbooru2021/filelist.txt.xz ./danboorumeta
unxz ./danboorumeta/filelist.txt.xz
rsync rsync://176.9.41.242:873/danbooru2021/metadata/posts*.json ./danboorumeta

# build container
docker-compose build
