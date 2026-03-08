#!/bin/bash
# Usage: run.sh <input_dir> <platform_dir> <output_dir> <top_module>

INPUT_DIR=$(realpath "$1")
PLATFORM_DIR=$(realpath "$2")
OUTPUT_DIR=$(realpath "$3")
TOP_MODULE="$4"
WORK_DIR=$(pwd)

export PYTHONHOME=$WORK_DIR/tools/my_env
export PYTHONPATH=$WORK_DIR/tools/my_env/lib/python3.13/site-packages:$WORK_DIR/tools/my_env/lib/python3.13
export LD_LIBRARY_PATH=$WORK_DIR/tools/my_env/lib:$LD_LIBRARY_PATH

source $WORK_DIR/tools/my_env/bin/activate

$WORK_DIR/tools/OpenROAD/build/bin/openroad -python flow.py \
    "$TOP_MODULE" \
    "$PLATFORM_DIR" \
    "$INPUT_DIR" \
    "$OUTPUT_DIR"
