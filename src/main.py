import logging
from tqdm import tqdm

import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import HfArgumentParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from llm import load_pipeline
from prompt import load_prompt
from arguments import ModelArguments, GenerationConfig, DataArguemnts, VectorDBArguments, RetrievalAtguments
from vectordb import VectorDB

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, GenerationConfig, DataArguemnts, VectorDBArguments, RetrievalAtguments))
    model_args, generation_config, data_args, vectordb_args, retrieval_args = parser.parse_args_into_dataclasses()

    llm = load_pipeline(model_args, generation_config)

    logging.info("Loading vector indexes from: %s", vectordb_args.index_path)
    vector_db = VectorDB(vectordb_args)
    vector_db.load()

    prompt = load_prompt()

    test = pd.read_csv(data_args.test_data, encoding="utf-8-sig")
    logging.info("Generating text for test data.. total rows: %d", len(test))

    test_results = []
    for idx, row in tqdm(test.iterrows(), total=len(test), desc="Generating responses"):
        category = row.get("category", "")
        query = row.get("query", "")

        retrieved_docs = vector_db.search(query, category, retrieval_args.k)
        train_examples = "\n\n".join(
            [
                f"사고원인: {retrieved_docs.page_content}\n방지대책: {retrieved_docs.metadata.solution}"
                for doc in retrieved_docs
            ]
        )

        chain = (
            {
                "train_examples": train_examples,
                "query": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(query=query)
        test_results.append(response)

        logging.info("Query   : %s", query)
        logging.info("Response: %s", response)

    logging.info("Genearation completed.. total results: %d", len(test_results))

    pred_embedding_model = SentenceTransformer("jhgan/ko-sbert-sts")
    pred_embeddings = pred_embedding_model.encode(test_results)

    submission = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")

    submission.iloc[:, 1] = test_results
    submission.iloc[:, 2:] = pred_embeddings

    submission.to_csv(data_args.submission_data, index=False, encoding="utf-8-sig")
    logging.info("Submission file saved at: %s", data_args.submission_data)


if __name__ == "__main__":
    main()
