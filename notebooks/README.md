# Data Preprocessing

## CSV [🔗](./data-preprocessing-csv.ipynb)

- **공사종류** : Split based on `/` to separate into `공사종류1`, `공사종류2`, `공사종류3`

- **공종** : Split based on `>` to separate into `공종1`, `공종2`

- **사고객체** : Split based on `>` to separate into `사고객체1`, `사고객체2`

- **장소** : Split based on `/` to separate into `장소1`, `장소2`

- **부위** : Split based on `/` to separate into `부위1`, `부위2`

- **인적사고** : Extract the string before `(` to separate into `인적사고1`

## PDF [🔗](./data-preprocessing-pdf.ipynb)

- Remove unnecessary pages such as cover pages, table of contents, etc.

- Split PDF documents into individual pages

- Extract text using `olmOCR` (execute `./pdf-test-olmocr.sh`)

    - Extract text from each separated page

    - Save results to the `./data-olmocr` directory

    - For detailed usage, refer to [allenai/olmocr](https://github.com/allenai/olmocr)

- Post-process `./data-olmocr`

    - Separate text using the `split_document_for_md` function

- Save final results to the `../data/documents` directory