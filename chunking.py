import re
import os
import pandas as pd
from typing import List, Dict, Any
import pymupdf4llm
from langchain.schema import Document


def parse_construction_type_csv(csv_path: str) -> Dict[str, str]:
    """
    pandas를 사용하여 CSV 파일에서 파일명과 공종 정보를 읽어와 매핑 딕셔너리를 생성합니다.

    Args:
        csv_path: CSV 파일 경로

    Returns:
        파일명을 키로, 공종을 값으로 하는 딕셔너리
    """
    try:
        # pandas로 CSV 파일 읽기
        df = pd.read_csv(csv_path, encoding="utf-8")

        # DataFrame을 딕셔너리로 변환
        file_to_construction_type = dict(zip(df["filename"].str.strip(), df["construction_type"].str.strip()))

        print(f"총 {len(file_to_construction_type)}개의 파일-공종 매핑 정보를 로드했습니다.")
        return file_to_construction_type

    except Exception as e:
        print(f"CSV 파일 읽기 중 오류 발생: {e}")
        # 오류 발생 시 빈 딕셔너리 반환
        return {}


def clean_text(text: str) -> str:
    """
    텍스트에서 특수 표시와 불필요한 형식을 제거합니다.
    """
    # ----- 줄 제거
    text = re.sub(r"\n*-----+\n*", "\n", text)

    # ``` 코드 블록 마커 제거 (내용은 유지)
    text = re.sub(r"```[\w]*\n", "", text)  # 시작 코드 블록 제거
    text = re.sub(r"\n```", "", text)  # 종료 코드 블록 제거

    # ***** 별표 표시 제거
    text = re.sub(r"\*+\d*\*+", "", text)

    # 연속된 빈 줄 제거
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def split_numbered_sections(text: str) -> List[Dict]:
    """
    텍스트에서 번호 체계를 기준으로 문서를 분할합니다.
    - 주요 섹션(1., 2. 등)과 하위 섹션(6.1, 6.2 등)을 모두 별도의 청크로 분할
    """
    # 텍스트 정리
    text = clean_text(text)

    # 줄바꿈 정규화
    text = text.replace("\r\n", "\n")

    # 두 가지 패턴을 정의: 주요 섹션 패턴(1., 2. 등)과 하위 섹션 패턴(6.1, 6.2 등)
    main_section_pattern = r"(?:^|\n)(\d+)\.\s+(.*?)(?=\n)"
    sub_section_pattern = r"(?:^|\n)(\d+\.\d+)\s+(.*?)(?=\n)"

    # 모든 주요 섹션과 하위 섹션 시작점 찾기
    main_section_matches = list(re.finditer(main_section_pattern, text))
    sub_section_matches = list(re.finditer(sub_section_pattern, text))

    # 모든 섹션 위치와 정보를 저장
    all_sections = []

    # 주요 섹션 정보 추가
    for match in main_section_matches:
        section_num = match.group(1)
        section_title = match.group(2).strip()
        start_pos = match.start()

        all_sections.append(
            {"type": "main", "number": section_num, "title": section_title, "start_pos": start_pos, "match": match}
        )

    # 하위 섹션 정보 추가
    for match in sub_section_matches:
        section_num = match.group(1)
        section_title = match.group(2).strip()
        start_pos = match.start()

        # 상위 섹션 번호 추출 (예: "6.1" -> "6")
        parent_num = section_num.split(".")[0]

        all_sections.append(
            {
                "type": "sub",
                "number": section_num,
                "title": section_title,
                "parent": parent_num,
                "start_pos": start_pos,
                "match": match,
            }
        )

    # 시작 위치로 정렬
    all_sections.sort(key=lambda x: x["start_pos"])

    if not all_sections:
        return []  # 섹션을 찾지 못했을 때 빈 리스트 반환

    # 각 섹션의 내용 추출
    result_sections = []

    for i, section in enumerate(all_sections):
        start_pos = section["start_pos"]

        # 다음 섹션의 시작 위치 찾기 (또는 문서 끝)
        if i < len(all_sections) - 1:
            end_pos = all_sections[i + 1]["start_pos"]
        else:
            end_pos = len(text)

        # 섹션 내용 추출
        content = text[start_pos:end_pos].strip()

        # 결과에 추가
        result_section = {
            "type": section["type"],
            "number": section["number"],
            "title": section["title"],
            "content": content,
        }

        # 하위 섹션인 경우 상위 섹션 정보 추가
        if section["type"] == "sub":
            result_section["parent"] = section["parent"]

        result_sections.append(result_section)

    return result_sections


def convert_to_langchain_documents(
    sections: List[Dict], pdf_path: str, construction_types: Dict[str, str]
) -> List[Document]:
    """
    섹션들을 LangChain Document 객체로 변환합니다.

    Args:
        sections: 분할된 섹션 리스트
        pdf_path: PDF 파일 경로
        construction_types: 파일명-공종 매핑 딕셔너리

    Returns:
        LangChain Document 객체 리스트
    """
    documents = []

    # 파일 이름과 경로 정보
    filename = os.path.basename(pdf_path)
    file_stem = os.path.splitext(filename)[0]

    # 해당 파일의 공종 정보 가져오기
    construction_type = construction_types.get(filename, "미분류")

    for section in sections:
        # 기본 메타데이터 구성
        metadata = {
            # 파일 정보
            "filename": filename,
            "title": file_stem,
            # 공종 정보 추가
            "construction_type": construction_type,
            # 섹션 정보
            "section_type": section["type"],  # "main" 또는 "sub"
            "section_number": section["number"],
            "section_title": section["title"],
            # 전체 경로 (파일명 > 섹션번호.제목)
            "path": f"{file_stem} > {section['number']}. {section['title']}",
        }

        # 하위 섹션인 경우 상위 섹션 정보 추가
        if section["type"] == "sub":
            metadata["parent_section"] = section["parent"]

            # 상위 섹션 제목 찾기
            parent_section_title = ""
            for s in sections:
                if s["type"] == "main" and s["number"] == section["parent"]:
                    parent_section_title = s["title"]
                    break

            if parent_section_title:
                metadata["parent_section_title"] = parent_section_title
                # 계층적 경로 추가
                metadata["hierarchical_path"] = (
                    f"{file_stem} > {section['parent']}. {parent_section_title} > {section['number']}. {section['title']}"
                )

        # Document 객체 생성
        doc = Document(page_content=section["content"], metadata=metadata)

        documents.append(doc)

    return documents


def process_pdf(pdf_path: str) -> List[Dict]:
    """
    PDF 파일을 처리하여 섹션별로 분할합니다.
    """
    # PDF를 마크다운으로 변환
    md_text = pymupdf4llm.to_markdown(pdf_path)

    # 섹션 분할
    return split_numbered_sections(md_text)


def save_documents_to_jsonl(documents: List[Document], output_file: str):
    """
    LangChain Document 객체를 JSONL 형식으로 저장합니다.

    Args:
        documents: LangChain Document 객체 리스트
        output_file: 저장할 파일 경로
    """
    import json

    with open(output_file, "w", encoding="utf-8") as f:
        for doc in documents:
            # Document 객체를 딕셔너리로 변환
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}

            # JSON 라인으로 변환하여 저장
            f.write(json.dumps(doc_dict, ensure_ascii=False) + "\n")

    print(f"{len(documents)}개의 Document 객체를 '{output_file}'에 JSONL 형식으로 저장했습니다.")


def process_all_pdfs(directory_path: str, construction_types_csv: str, output_dir: str = None) -> Dict:
    """
    지정된 디렉토리에 있는 모든 PDF 파일을 처리합니다.

    Args:
        directory_path: PDF 파일이 있는 디렉토리 경로
        construction_types_csv: 파일명-공종 매핑 CSV 파일 경로
        output_dir: 처리 결과를 저장할 디렉토리 (선택 사항)

    Returns:
        처리 결과 요약 딕셔너리
    """
    # 결과 저장 디렉토리 생성 (지정된 경우)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 공종 정보 로드
    construction_types = parse_construction_type_csv(construction_types_csv)

    # 디렉토리 내 모든 PDF 파일 목록 가져오기
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]

    print(f"총 {len(pdf_files)}개의 PDF 파일을 발견했습니다.")

    # 처리 결과 통계
    successful_files = 0
    failed_files = []
    empty_files = []  # 섹션이 없는 파일
    total_documents = 0  # 총 생성된 Document 수
    missing_construction_type = []  # 공종 정보가 없는 파일

    # 각 PDF 파일 처리
    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(directory_path, pdf_file)
        print(f"[{i + 1}/{len(pdf_files)}] 처리 중: {pdf_file}")

        # 공종 정보 확인
        if pdf_file not in construction_types:
            print(f"  - 경고: '{pdf_file}'의 공종 정보가 없습니다.")
            missing_construction_type.append(pdf_file)

        try:
            # PDF 처리
            sections = process_pdf(pdf_path)

            # 섹션이 없는 경우
            if not sections:
                print(f"  - 경고: '{pdf_file}'에서 섹션을 찾을 수 없습니다.")
                empty_files.append(pdf_file)
                continue

            # LangChain Document 객체로 변환
            documents = convert_to_langchain_documents(sections, pdf_path, construction_types)
            total_documents += len(documents)

            # 처리 결과 출력
            main_sections = [s for s in sections if s["type"] == "main"]
            sub_sections = [s for s in sections if s["type"] == "sub"]
            print(f"  - 성공: 주요 섹션 {len(main_sections)}개, 하위 섹션 {len(sub_sections)}개 발견")
            print(f"  - LangChain Document {len(documents)}개 생성")
            print(f"  - 공종: {construction_types.get(pdf_file, '미분류')}")

            # 각 문서별로 JSONL 파일 저장
            if output_dir:
                jsonl_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}.jsonl")
                save_documents_to_jsonl(documents, jsonl_file)

            successful_files += 1

        except Exception as e:
            print(f"  - 오류: '{pdf_file}' 처리 중 예외 발생: {str(e)}")
            failed_files.append((pdf_file, str(e)))

    # 처리 결과 요약
    print("\n" + "=" * 50)
    print("처리 완료!")
    print(f"- 성공: {successful_files}개 파일")
    print(f"- 실패: {len(failed_files)}개 파일")
    print(f"- 섹션 없음: {len(empty_files)}개 파일")
    print(f"- 공종 정보 없음: {len(missing_construction_type)}개 파일")
    print(f"- 총 청크 수: {total_documents}개")

    # 실패한 파일 목록 출력
    if failed_files:
        print("\n실패한 파일 목록:")
        for failed_file, error in failed_files:
            print(f"- {failed_file}: {error}")

    # 섹션이 없는 파일 목록 출력
    if empty_files:
        print("\n섹션이 없는 파일 목록:")
        for empty_file in empty_files:
            print(f"- {empty_file}")

    # 공종 정보가 없는 파일 목록 출력
    if missing_construction_type:
        print("\n공종 정보가 없는 파일 목록:")
        for missing_file in missing_construction_type:
            print(f"- {missing_file}")

    return {
        "successful_files": successful_files,
        "failed_files": failed_files,
        "empty_files": empty_files,
        "missing_construction_type": missing_construction_type,
        "total_documents": total_documents,
    }


def main():
    directory_path = "./data/건설안전지침"
    output_dir = "./data/건설안전지침_분할결과"

    construction_types_csv = "./data/file_construction_type_analysis.csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"결과 저장 디렉토리 '{output_dir}'를 생성했습니다.")

    process_all_pdfs(directory_path, construction_types_csv, output_dir)


if __name__ == "__main__":
    main()
