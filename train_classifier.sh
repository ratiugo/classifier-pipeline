#!/bin/bash

config_file_name=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --classifier-config-file-name)
            config_file_name=$2
            shift 2
            ;;

        *)
            break
            ;;
    esac
done

echo $config_file_name

PYTHONPATH=. python 


PYTHONPATH=. python src/train_classifier.py \
    --data-file-name $data_file_name \
    --classifier-config-file-name $config_file_name
