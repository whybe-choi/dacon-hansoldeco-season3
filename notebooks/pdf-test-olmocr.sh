PDF_DIR="./pdf-extract-pages"
OUTPUT_DIR="./data-olmocr"

# pdf_001부터 pdf_104까지의 폴더 순차적으로 처리
for i in $(seq -f "%03g" 1 104); do
    folder="pdf_$i"
    echo "폴더 처리 중: $folder"
    
    # 폴더가 존재하는지 확인
    if [ -d "$PDF_DIR/$folder" ]; then
        # 출력 디렉토리 생성
        mkdir -p "$OUTPUT_DIR/$folder"
        
        # olmocr 파이프라인 실행
        echo "처리 중: $PDF_DIR/$folder"
        python -m olmocr.pipeline "$OUTPUT_DIR/$folder" --pdfs "$PDF_DIR/$folder"/*.pdf
    else
        echo "경고: $PDF_DIR/$folder 폴더가 존재하지 않습니다."
    fi
done