python code/preprocess_data.py \
    --dataset_path "data/combine_data/ram/combine.csv" \
    --save_folder_path "data/preprocess_data/ram" \
    --ratio 0.8 \
    --group "H"

python code/preprocess_data.py \
    --dataset_path "data/combine_data/cpu/combine.csv" \
    --save_folder_path "data/preprocess_data/cpu" \
    --ratio 0.8 \
    --group "H"