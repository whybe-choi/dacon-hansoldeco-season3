import os
import json
from typing import List

from langchain_core.documents import Document


def load_documents(documents_path: str) -> List[Document]:
    file_list = sorted(os.listdir(documents_path))
    documents = []
    for file in file_list:
        if file.endswith(".jsonl"):
            with open(f"{documents_path}/{file}", "r", encoding="utf-8-sig") as f:
                for line in f:
                    line_data = json.loads(line)
                    if line_data["metadata"]["type"] != "intro":
                        documents.append(
                            Document(
                                page_content=line_data["page_content"],
                                metadata=line_data["metadata"],
                            )
                        )

    return documents


def load_query_expansions(query_expansions_path: str) -> List[str]:
    with open(query_expansions_path, "r", encoding="utf-8-sig") as f:
        query_expansions = []
        for line in f:
            data = json.loads(line)
            query_expansions.append(data)

    return query_expansions


def format_references(docs):
    return "\n\n".join([doc for doc in docs])


def format_qa_pairs(doc_pairs):
    qa_pair_texts = ""
    for doc_pair in doc_pairs:
        qa_pair_texts += f"Q. {doc_pair['question']}\nA. {doc_pair['response']}\n\n"
    return qa_pair_texts
