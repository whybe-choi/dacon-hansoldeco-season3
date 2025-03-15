import os
from typing import List
from tqdm import tqdm

from transformers import HfArgumentParser
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import models

from arguments import VectorDBArguments


class VectorDB:
    def __init__(self, args: VectorDBArguments):
        self.args = args
        self.embedding = HuggingFaceEmbeddings(model_name=args.embedding_model)
        self.vector_store = None

    def _load_train_dataset(self) -> List[Document]:
        documents = []
        if not os.path.exists(self.args.train_data):
            raise FileNotFoundError(f"Dataset file not found: {self.args.train_data}")

        train_dataset = load_dataset("csv", data_files=self.args.train_data, split="train")

        for example in tqdm(train_dataset, desc="Train Dataset Loading.."):
            cause = example.get("사고원인", "")
            if not cause:
                continue
            document = Document(
                page_content=example["사고원인"],
                metadata={
                    "category": example.get("공종(중분류)", "Unknown"),
                    "solution": example.get("재발방지대책 및 향후조치계획", ""),
                },
            )
            documents.append(document)

        return documents

    def build(self):
        os.makedirs(self.args.index_path, exist_ok=True)

        train_documents = self._load_train_dataset()

        if train_documents:
            self.train_store = QdrantVectorStore.from_documents(
                documents=train_documents,
                embedding=self.embedding,
                path=self.args.index_path,
                collection_name="train",
            )
            print(f"Train Dataset Index saved on: {self.args.index_path}")

    def load(self):
        if os.path.exists(self.args.index_path):
            self.train_store = QdrantVectorStore.from_existing_collection(
                embedding=self.embedding,
                collection_name="train",
                path=self.args.index_path,
            )
            print(f"Train vector database loaded from {self.args.index_path}")
        else:
            print("Warning: Train index not found.")

    def search(self, query: str, category: str, k: int = 3) -> List[Document]:
        if not self.train_store:
            raise ValueError("Train store not loaded.")

        return self.train_store.similarity_search(
            query=query,
            k=k,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.category",
                        match=models.MatchValue(value=category),
                    ),
                ]
            ),
        )


def main():
    parser = HfArgumentParser(VectorDBArguments)
    args = parser.parse_args_into_dataclasses()[0]

    vector_db = VectorDB(args)
    vector_db.build()


if __name__ == "__main__":
    main()
