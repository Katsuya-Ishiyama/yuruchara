#!/bin/sh

work_dir=/home/ishiyama/yuruchara

python ${work_dir}/download_images.py ${work_dir} company
python ${work_dir}/download_images.py ${work_dir} gotochi

tar -zcvf yuruchara.tar.gz ${work_dir}

