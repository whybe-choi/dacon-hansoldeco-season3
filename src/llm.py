import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from accelerate import Accelerator

from langchain_huggingface.llms import HuggingFacePipeline


def load_pipeline(model_args, generation_config):
    if model_args.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config if model_args.use_bnb else None,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        token=model_args.token,
    )

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=generation_config.do_sample,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        return_full_text=generation_config.return_full_text,
        max_new_tokens=generation_config.max_new_tokens,
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return llm
