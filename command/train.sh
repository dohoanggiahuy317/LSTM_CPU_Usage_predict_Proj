python code/train.py \
    --train_path "data/preprocess_data/ram/train.csv" \
    --save_model_path "model/ram/12H" \
    --prev_day 12 \
    --pred_day 2 \
    --epochs 5 \
    --batch_size 128 \
    --verbose 1 \
    --shuffle false \


python code/train.py \
    --train_path "data/preprocess_data/cpu/train.csv" \
    --save_model_path "model/cpu/12H" \
    --prev_day 12 \
    --pred_day 2 \
    --epochs 5 \
    --batch_size 128 \
    --verbose 1 \
    --shuffle false \