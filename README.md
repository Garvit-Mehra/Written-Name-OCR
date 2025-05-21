# BiLSTM-CRNN OCR for Handwritten Name Recognition

This project implements an Optical Character Recognition (OCR) model designed to recognize handwritten names using a Convolutional Recurrent Neural Network (CRNN) with BiLSTM layers and CTC loss. It is trained on grayscale name images and corresponding labels using TensorFlow.

⸻

📑 Table of Contents
  1. Overview
	2.	Model Architecture
	3.	Installation
	4.	Dataset
	5.	Usage
	6.	Evaluation
	7.	Results
	8.	Acknowledgements

---

🧩 Overview

This project builds an end-to-end OCR model to transcribe images of handwritten names into digital text. It uses:
- CNN layers for spatial feature extraction
- BiLSTM layers to learn sequential dependencies
- CTC Loss for alignment-free sequence prediction

---

🧱 Model Architecture
	1.	CNN Feature Extractor
  - Multiple Conv2D + MaxPooling layers
  - BatchNormalization & Dropout for generalization
	2.	BiLSTM Decoder
  - Bidirectional LSTM layers capture temporal patterns in sequences.
	3.	CTC Loss Function
  - Allows the model to learn sequences without needing exact alignments.

---

⚙️ Installation
```bash
git clone https://github.com/yourusername/bilstm-ocr
cd bilstm-ocr
pip install -r requirements.txt
```
Dependencies:
	•	Python 3.x
	•	TensorFlow
	•	NumPy, Pandas, OpenCV, scikit-learn, Matplotlib

---

📁 Dataset

The dataset includes:
- CSV files with image filenames and labels (written_name_train.csv, written_name_valid.csv)
- Image folders:
- data/train/train/
- data/valid/valid/

Note: The model filters out samples labeled as "UNREADABLE".

---

🚀 Usage

1. Preprocessing
  - Converts grayscale images to fixed size (64×256)
  - Rotates images and normalizes pixel values

2. Training the Model
  - Uses early stopping and model checkpointing
  - Hyperparameters like IMAGE_HEIGHT, SEQ_LENGTH, and ALPHABET are customizable

3. Prediction
  - The trained model can be used to infer labels from unseen handwritten images.

📊 Evaluation

Metrics:
  - Character Accuracy
  - Word Accuracy

Predicted sequences are compared with ground truth using CTC decoding.

---

🙌 Acknowledgements
- Inspired by academic OCR literature and practical CRNN designs
- Built using TensorFlow and OpenCV
- Dataset inspired by handwritten name datasets like IAM or custom-collected datasets
