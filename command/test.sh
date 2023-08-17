python code/test.py \
    --eval_path "data/test_data/ram/ram_9aug_16aug.csv" \
    --model_path "model/ram/12H" \
    --save_pred_path "Result/12H/ram/result.csv" \
    --prev_day 12 \
    --pred_day 2 \
    --group "H"

python code/test.py \
    --eval_path "data/test_data/cpu/cpu_9aug_16aug.csv" \
    --model_path "model/cpu/12H" \
    --save_pred_path "Result/12H/cpu/result.csv" \
    --prev_day 12 \
    --pred_day 2 \
    --group "H"
