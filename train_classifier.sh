#!/bin/bash

data_file_name=""
classifier_config_file_name=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-file-name)
            data_file_name=$2
            shift 2
            ;;

        --classifier-config-file-name)
            classifier_config_file_name=$2
            shift 2
            ;;

        *)
            break
            ;;
    esac
done

echo $data_file_name
echo $classifier_config_file_name

PYTHONPATH=. python src/train_classifier.py \
    --data-file-name $data_file_name \
    --classifier-config-file-name $classifier_config_file_name
