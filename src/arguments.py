import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_bnb: Optional[bool] = field(default=False, metadata={"help": "Whether to use BitsAndBytes."})
    token: Optional[str] = field(default="")


@dataclass
class GenerationConfig:
    do_sample: Optional[bool] = field(default=False, metadata={"help": "Whether to use sampling."})
    temperature: Optional[float] = field(default=None, metadata={"help": "Sampling temperature."})
    top_p: Optional[float] = field(default=None, metadata={"help": "Top-p sampling."})
    max_new_tokens: Optional[int] = field(default=384, metadata={"help": "Max number of tokens to generate."})
    return_full_text: Optional[bool] = field(default=False, metadata={"help": "Whether to return full text."})


@dataclass
class DataArguemnts:
    test_data: str = field(metadata={"help": "Path to test"})
    submission_data: str = field(metadata={"help": "Path to submission"})

    def __post_init__(self):
        if not os.path.exists(self.test_data):
            raise FileNotFoundError(f"Dataset file not found: {self.test_data}")


@dataclass
class VectorDBArguments:
    embedding_model: str = field(metadata={"help": "Path to embedding model for indexing"})
    train_data: Optional[str] = field(default=None, metadata={"help": "Path to train data"})
    index_path: Optional[str] = field(default="./qdrant", metadata={"help": "Path to save/load vector db"})


@dataclass
class RetrievalAtguments:
    top_k: int = field(default=5, metadata={"help": "Top k for RAG})"})
    use_reranker: Optional[bool] = field(default=False, metadata={"help": "Whether to use reranker."})
    reranker_model: Optional[str] = field(default="", metadata={"help": "Path to reranker model."})
    reranker_top_k: Optional[int] = field(default=5, metadata={"help": "Top k for reranker."})
