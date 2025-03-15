from langchain_core.prompts import PromptTemplate


def load_prompt():
    prompt_template = """<INSTRUCTION>
You are a construction safety expert. \
Ensure that you use the provided context information to generate your answer. \
Use one sentence maximum and keep the answer concise. \
Do not include any additional explanations (such as introductions or background information). \
Be sure to answer in Korean.
</INSTRUCTION>

<EXAMPLES>
{train_examples}
</EXAMPLES>

Question: {query}
Answer:"""

    prompt = PromptTemplate(
        input_variables=["train_examples", "query"],
        template=prompt_template,
    )

    return prompt
