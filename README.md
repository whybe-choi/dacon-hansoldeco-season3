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
## Chunking
```bash
python chunking.py
```

## Indexing

```bash
python src/vectordb.py \
    --embedding_model nlpai-lab/KURE-v1 \
    --train_data ./data/train_preprocessed.csv \
    --document_path ./data/건설안전지침_분할결과 \
    --index_path ./faiss
```