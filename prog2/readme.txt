# README for proj2.py

## Overview
`proj2.py` is a Python script designed for training, evaluating, and running inference on text data using various deep learning models. The script supports multiple model architectures and provides options for data preprocessing, training, and inference.

## Usage
To run the script, use the following command:
python proj2.py --tokenizer_path <path_to_tokenizer> --data_dir <path_to_data> --output_dir <path_to_output> --model_option <model_type> --num_layers <num_layers> --max_seq_length <max_seq_length> --batch_size <batch_size> --epochs <num_epochs> --learning_rate <learning_rate> --weight_decay <weight_decay> --patience <patience> [--test_only] [--recreate_data_and_tokenizer] [--inference_only "<inference_text>"]

The script assumes that the model files are in the working directory.

The easiest way to test the model is by doing 
python proj2.py --model_option Transformer --prompt "hello world" --temperature 1.0


##FYI
Temperature works I'm pretty sure, but since the perplexity is so high it doesn't really affect much.