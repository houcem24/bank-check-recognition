Autocorrection System for Bank Checks

(Deep Learning & Computer Vision Project)

This project develops an autocorrection / auto-recognition system for bank checks (cheques), using deep learning and computer vision techniques. The system aims to detect, extract, and correct information from scanned or photographed bank check images (e.g., payer name, amount, account number, date), combining object detection and optical character recognition (OCR).

🎯 Motivation & Scope

Manually processing bank checks is time-consuming and error-prone. Automating the extraction and error correction of check fields can significantly improve efficiency and accuracy in banking systems.

This project integrates:

Object detection (to localize the individual fields on a check)

OCR (to read text from detected regions)

Post-processing / correction steps to fix recognition errors

📂 Repository Structure

Below is a suggested / observed layout (modify to reflect your actual structure):

.
├── data/                  # check image dataset, annotations, test splits  
├── models/                # saved model weights (object detection, OCR)  
├── notebooks/             # training, evaluation, exploration notebooks  
├── src/                   # source code (detection, OCR pipeline, utils)  
│   ├── detection.py  
│   ├── ocr.py  
│   ├── correction.py  
│   └── inference.py  
├── requirements.txt  
├── README.md  
├── LICENSE  
└── demo/                  # sample inputs & outputs (images, corrected text)  


You can update this structure based on your actual repository.

🛠️ Key Components & Workflow

Data Preparation & Annotation

Collect scanned / photographed check images

Annotate bounding boxes for fields (e.g. name, date, amount)

Object Detection Model

A deep learning model (e.g. YOLO family or Faster R-CNN) to locate field regions

OCR Module

For each detected field, apply OCR (e.g. Tesseract, EasyOCR, or custom network)

Preprocess region (cropping, thresholding) to improve recognition

Post-processing & Correction

Clean OCR text: remove noise, fix known formatting

Validate numeric fields (e.g. amount) against rules

Spell check / dictionary correction for names

Inference Pipeline

Given a new check image → detect fields → OCR → correct → output structured result

Evaluation

Metrics: detection IoU, OCR character accuracy, field-level accuracy

Visual results: bounding box overlays, text overlays

⚙️ Setup & Installation

Clone the repository:

git clone https://github.com/houcem24/Deep-Learning-and-Computer-Vision-Project-Autocorrection-System-of-Bank-checks.git
cd Deep-Learning-and-Computer-Vision-Project-Autocorrection-System-of-Bank-checks


Install dependencies (preferably in a virtual environment):

pip install -r requirements.txt


Download / prepare datasets and annotation files, and place them in the data/ folder.

(If required) download pretrained model weights and place in models/.

🚀 Usage
Training

You can train detection / OCR models using scripts or notebooks. For example:

python src/train_detection.py --config configs/detect_config.yaml
python src/train_ocr.py --config configs/ocr_config.yaml

Inference / Demo

Run the inference pipeline on new check images:

python src/inference.py --input_path demo/sample_check.jpg --output_path demo/output.json


You should get a JSON (or other structured) output containing the detected fields and corrected text.

You can also visualize bounding boxes and recognized text region overlays for inspection.

📊 Evaluation & Results

Object detection performance (e.g. mean IoU, detection accuracy)

OCR accuracy / character error rate

Field-level accuracy (correct full field vs partial)

Examples & visualizations (before vs after correction)

Include sample images / result visuals in demo/ or in the README.

🔮 Future Work & Improvements

Expand dataset diversity (various check templates, lighting conditions)

Use more advanced detection networks (e.g. transformer-based)

Fine-tune OCR models for handwriting / stylized fonts

Improve error correction using contextual / language models

Build a GUI or web interface for uploading checks

Real-time processing / mobile deployment
