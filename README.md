# Bank Check Amount Verification

A computer vision pipeline that reads a scanned bank check, recognises both the amount written in digits
(courtesy amount) and the amount written in words (legal amount), corrects spelling errors in the handwriting
recognition, and flags checks where the two amounts do not match.

## Pipeline

1. **Preprocessing (OpenCV):** grayscale conversion, binary thresholding, morphological filtering and Hough-line
   detection.
2. **Printed text (Tesseract OCR):** bank name, branch and other printed fields.
3. **Segmentation:** dilation and contour detection crop the courtesy-amount box into single digits and the
   legal-amount line into single words.
4. **Courtesy amount:** a CNN trained on MNIST-style 28×28 digits classifies each digit
   (`digits_recognition.h5`).
5. **Legal amount:** words are read with a CNN-RNN-CTC handwriting model based on
   [SimpleHTR](https://github.com/githubharald/SimpleHTR), trained on the IAM dataset.
6. **Auto-correction:** recognised words are corrected against the vocabulary of amounts
   (one, two, … thousand, lakh) with `pyspellchecker`. An experimental sequence model for the same task
   (`autocorrection.h5`) is loaded at the end of the notebook.
7. **Verification:** the digit amount is converted to words (Indian numbering: lakh, thousand, hundred) and
   compared with the corrected legal amount.

Example from the notebook: misread words such as `latch`, `Twentrh` and `Thrusond` are corrected to
"one lakh twenty five thousand", and the check is flagged because the digits read 492,399.

## Repository structure

```
├── bank_check_recognition.ipynb   # End-to-end pipeline
├── requirements.txt
└── docs/                          # Course deliverables: technical and sales presentations,
                                   # project report, poster, business model canvas
```

## Running the notebook

```bash
git clone https://github.com/houcem24/bank-check-recognition.git
cd bank-check-recognition
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The notebook also needs:

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on your `PATH`
- a checkout of [SimpleHTR](https://github.com/githubharald/SimpleHTR) with its pretrained model
- the trained models `digits_recognition.h5` and `autocorrection.h5` and a sample check image

The trained models and check images are not included in this repository. Put them in one folder and point the
notebook at it with `export BANK_CHECK_DATA_DIR=/path/to/folder`.

## Context

Integrated data science project at ESPRIT (2022), built in a team of six: Youssef Mansour, Molka Mrad,
Houssem Ouerdiane, Dhia Ben Amor, Haithem Gharbi and Aicha Chaaba.

## License

[MIT](LICENSE). SimpleHTR is © Harald Scheidl under its own MIT license.
