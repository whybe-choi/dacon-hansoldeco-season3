import logging
from tqdm import tqdm

import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import HfArgumentParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.output_parsers import StrOutputParser
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from llm import load_pipeline
from prompt import load_reference_prompt, load_rag_prompt
from arguments import ModelArguments, GenerationConfig, DataArguments, RetrievalArguments
from utils import load_documents, load_query_expansions, format_references, format_qa_pairs

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, GenerationConfig, DataArguments, RetrievalArguments))
    model_args, generation_config, data_args, retrieval_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    test = pd.read_csv(data_args.test_data, encoding="utf-8-sig")

    llm = load_pipeline(model_args, generation_config)
    reference_prompt = load_reference_prompt()
    rag_prompt = load_rag_prompt()

    query_expansions = load_query_expansions(data_args.query_expansions_path)
    documents = load_documents(data_args.documents_path)

    # 문서를 위한 벡터 저장소 생성
    vector_store = FAISS.from_documents(
        documents=documents, embedding=HuggingFaceEmbeddings(model_name=retrieval_args.embedding_model)
    )
    base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": retrieval_args.top_k})

    # Reranker 모델을 사용하여 검색 결과를 재정렬
    model = HuggingFaceCrossEncoder(model_name=retrieval_args.reranker_model)
    compressor = CrossEncoderReranker(model=model, top_n=retrieval_args.reranker_top_k)
    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

    ###########################################################
    # Step1 : 개별 테스트 데이터에 대한 다수의 쿼리 확장에 대한 답변을 생성 #
    ###########################################################
    reference_chain = reference_prompt | llm | StrOutputParser()

    logging.info("Generating references for query expansions.. total expansions: %d", len(query_expansions))
    references = []
    for idx, query_expansion in tqdm(enumerate(query_expansions), total=len(query_expansions), desc="Generating references"):
        test_id = query_expansion["test_id"]
        questions = query_expansion["questions"]

        logging.info(f"[{idx + 1}/{len(query_expansions)}  {test_id}]")

        reference = []
        for q_idx, question in enumerate(questions):
            context = retriever.invoke(question)
            context = format_references(context)
            response = reference_chain.invoke({"question": question, "context": context})
            response = response.strip()

            logging.info("[Q %d]", q_idx)
            logging.info("Question: %s", question)
            logging.info("Response: %s", response)

            reference.append({"question": question, "response": response})
        references.append({"test_id": test_id, "references": reference})

    ##################################################################
    # Step2 : 확장된 쿼리와 그에 대한 답변을 통해 사고 원인에 대한 사고방지 대책 생성 #
    ##################################################################

    rag_chain = rag_prompt | llm | StrOutputParser()

    test = pd.read_csv(data_args.test_data, encoding="utf-8-sig")
    logging.info("Generating text for test data.. total rows: %d", len(test))

    test_results = []
    for idx, row in tqdm(test.iterrows(), total=len(test), desc="Generating responses"):
        query = row["사고원인"]
        context = references[idx]
        context = format_qa_pairs(context["references"])

        response = rag_chain.invoke({"query": query, "context": context})

        test_results.append(response)

        logging.info("[Row %d]", idx)
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
