# dacon-hansoldeco-season3
<img width="1190" alt="Image" src="https://github.com/user-attachments/assets/6904a0cc-5b78-4a8f-ba38-a86cf33d56e7" />

## Results
|Name|Type|Performance|Rank|
|---|---|---|---|
|**[건설공사 사고 예방 및 대응책 생성 : 한솔데코 시즌3 AI 경진대회](https://dacon.io/competitions/official/236455/overview/description)**|NLP, LLM|||

## Environment

```bash
conda create -n dacon python=3.10
conda activate dacon
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Methodology
For a detailed explanation of the methodology, please refer to our [presentation slides](./slides/[Team%20YG]%20데이콘_한솔데코3.pdf)

![Image](https://github.com/user-attachments/assets/40a8026a-a1ad-4f6d-95aa-4d90e5eb5ec7)

## Preprocessing

Please use **[this notebook](./notebooks/data-preprocessing-csv.ipynb)** for data preprocessing

## Query Expansion
```bash
python src/expand_query.py \
    --model_name_or_path rtzr/ko-gemma-2-9b-it \
    --test_data ./data/test_preprocessed.csv \
    --output_data ./data/query_expansions.jsonl \
    --token YOUR_HF_TOKEN
```

## RAG
```bash
python src/main.py \
    --model_name_or_path rtzr/ko-gemma-2-9b-it \
    --attn_implementation eager \
    --test_data ./data/test_preprocessed.csv \
    --query_expansions_path ./data/query_expansions.jsonl \
    --embedding_model nlpai-lab/KURE-v1 \
    --top_k 30 \
    --documents_path ./data/documents \
    --use_reranker true \
    --reranker_model dragonkue/bge-reranker-v2-m3-ko \
    --reranker_top_k 10 \
    --submission_data ./submissions/submission.csv \
    --token YOUR_HF_TOKEN
```

## Members
|홍재민|최용빈|
| :-: | :-: |
| <a href="https://github.com/geminii01" target="_blank"><img src='https://avatars.githubusercontent.com/u/171089104?v=4' height=130 width=130></img> | <a href="https://github.com/whybe-choi" target="_blank"><img src='https://avatars.githubusercontent.com/u/64704608?v=4' height=130 width=130></img> |
| <a href="https://github.com/geminii01" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a> | <a href="https://github.com/whybe-choi" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a> |
