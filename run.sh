CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")

python src/main.py \
    --model_name_or_path rtzr/ko-gemma-2-9b-it \
    --test_data ./data/test_preprocessed.csv \
    --embedding_model nlpai-lab/KURE-v1 \
    --index_path ./qdrant \
    --top_k 6 \
    --submission_data ./submissions/submssion_${CURRENT_TIME}.csv