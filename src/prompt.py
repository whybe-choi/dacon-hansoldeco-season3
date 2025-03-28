from langchain_core.prompts import PromptTemplate


def load_reference_prompt():
    prompt_template = """You are a construction safety expert. \
Based on the provided question and retrieved safety guidelines, generate a single, comprehensive preventive measure that incorporates key points from the guidelines. \
Follow these guidelines when responding:
- Do not provide multiple alternative measures—combine key ideas into one well-structured sentence.
- Ensure the measure is clear, actionable, and directly applicable to construction sites.
- Answer must be within ONLY ONE sentence in KOREAN.

Question: {question}
Context:
{context}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["question", "context"],
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


def load_rag_prompt():
    prompt_template = """You are a construction safety expert. \
Your task is  to answer preventive measures for the cause of an incident given as a query based on context.

The following responses were generated for different aspects of the same safety concern. Each response addresses a specific question related to the main safety issue.

Follow these guidelines when responding:
- Identify the core safety concern across all questions and answers
- Extract and combine the most critical safety measures from all responses
- Create a unified, comprehensive safety guideline that addresses the central issue
- Ensure the final answer is clear, actionable, and directly applicable to construction sites
- Answer must be within ONLY ONE sentence in KOREAN.

Query: {query}

Context:
{context}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["query, context"],
        template=prompt_template,
    )

    return prompt
