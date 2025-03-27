# dacon-hansoldeco-season3
<img width="1190" alt="Image" src="https://github.com/user-attachments/assets/04c25ede-982d-4ccb-94cf-ef0e1ca9434d" />

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

## Indexing

```bash
python src/vectordb.py \
    --embedding_model nlpai-lab/KURE-v1 \
    --train_data ./data/train_preprocessed.csv \
    --index_path ./qdrant
```

## Query Expansion
```bash
python src/expand_query.py \
    --model_name_or_path HumanF-MarkrAI/Gukbap-Gemma2-9B \
    --test_data ./data/test_preprocessed.csv \
    --output_data ./data/query_expansions.jsonl \
    --token YOUR_HF_TOKEN
```

## RAG
```bash
python src/main.py \
    --model_name_or_path rtzr/ko-gemma-2-9b-it \
    --test_data ./data/test_preprocessed.csv \
    --embedding_model nlpai-lab/KURE-v1 \
    --index_path ./qdrant \
    --top_k 5 \
    --submission_data ./submissions/submssion.csv
    --token YOUR_HF_TOKEN
```

## Members
|홍재민|최용빈|
| :-: | :-: |
| <a href="https://github.com/geminii01" target="_blank"><img src='https://avatars.githubusercontent.com/u/171089104?v=4' height=130 width=130></img> | <a href="https://github.com/whybe-choi" target="_blank"><img src='https://avatars.githubusercontent.com/u/64704608?v=4' height=130 width=130></img> |
| <a href="https://github.com/geminii01" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a> | <a href="https://github.com/whybe-choi" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a> |
