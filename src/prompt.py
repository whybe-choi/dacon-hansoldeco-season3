from langchain_core.prompts import PromptTemplate


def load_rag_prompt():
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


def load_query_expansion_prompt():
    prompt_template = """You are a construction safety expert. \
Your task is to generate focused questions that will help identify preventive measures from safety guideline documents based on given accident scenarios. \
Each question should be concise, clear, and focused on a single topic or perspective. \
These questions will be used to search through safety documentation to find relevant preventive measures and protocols. \
Avoid compound questions or questions that address multiple scenarios simultaneously. \
Please generate 3 questions in Korean.

While performing '{job_process}' at '{gongjong}', an incident involving '{human_accident}' occurred at '{accident_object}'. \
The cause of the accident is as follows: \
'{accident_cause}'

We are trying to find preventive measures in the safety guideline documents. Please generate 3 questions in Korean.

<Output Format>
{{
    "questions": [
        "Question 1",
        "Question 2",
        "Question 3"
    ]
}}
"""
    prompt = PromptTemplate(
        input_variables=["job_process", "gongjong", "human_accident", "accident_object", "accident_cause"],
        template=prompt_template,
    )

    return prompt
