from langchain_core.prompts import PromptTemplate


def load_prompt():
    prompt_template = (
        "당신은 건설 안전 전문가입니다.\n"
        "사고원인에 대한 방지대책을 핵심 내용만 요약하여 간략하게 작성하세요.\n"
        "서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.\n"
        "다음과 같은 조치를 취할 것을 제안합니다: 와 같은 내용을 포함하지 마세요.\n"
        "\n"
        "{train_examples}"
        "\n"
        "사고원인: {query}\n"
        "방지대책: "
    )

    prompt = PromptTemplate(
        input_variables=["train_examples", "query"],
        template=prompt_template,
    )

    return prompt
