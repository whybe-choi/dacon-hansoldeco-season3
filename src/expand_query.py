import logging
from tqdm import tqdm
import json

import pandas as pd
from transformers import HfArgumentParser
from langchain_core.output_parsers import StrOutputParser

from llm import load_pipeline
from prompt import load_query_expansion_prompt
from arguments import ModelArguments, GenerationConfig, DataArguments, RetrievalArguments

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, GenerationConfig, DataArguments, RetrievalArguments))
    model_args, generation_config, data_args, retrieval_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    llm = load_pipeline(model_args, generation_config)
    prompt = load_query_expansion_prompt()

    test = pd.read_csv(data_args.test_data, encoding="utf-8-sig")

    chain = prompt | llm | StrOutputParser()

    results = []
    for i, row in tqdm(test.iterrows(), total=len(test), desc="Generating expanded queries"):
        kwargs = {
            "gongjong": row["공종2"],
            "job_process": row["작업프로세스"],
            "accident_object": row["사고객체1"] + ", " + row["사고객체2"],
            "human_accident": row["인적사고1"],
            "accident_cause": row["사고원인"],
        }

        # 체인 실행
        response = chain.invoke(kwargs)
        response = response.replace("model```json\n", "")
        response = response.replace("```", "")
        response_json = json.loads(response)

        result = {"quesitons": response_json.get("quesitons", []), "test_id": row["ID"]}
        results.append(result)
        print(result)

    with open(data_args.output_data, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
