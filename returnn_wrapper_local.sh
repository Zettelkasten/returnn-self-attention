#!/usr/bin/env bash 

export OMP_NUM_THREADS=3 
export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.1/extras/CUPTI/lib64:/usr/local/cudnn-10.1-v7.6/lib64:/usr/lib/nvidia-418" 
export PYTHONPATH=/u/petrick/software/returnn/official/:/u/petrick/software/returnn/official/tests/:$PYTHONPATH

/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python "$@" 
