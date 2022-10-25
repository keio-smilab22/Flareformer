#!/bin/bash -eu

# set root directory
root_dir=$(readlink -f $(dirname $0)/../)
cd $root_dir

# start server on poetry
poetry run python src/flareformer/main.py --mode=server --params=config/params_2017.json
