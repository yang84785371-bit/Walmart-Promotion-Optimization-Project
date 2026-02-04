#!/usr/bin/env bash
set -e

cd /home/didu/projects
source .venv/bin/activate

cd walmart_sales_project

DATA_DIR=/home/didu/projects/datasets/walmart_sales_data
python src/step1_build_dataset.py --data_dir "$DATA_DIR"
