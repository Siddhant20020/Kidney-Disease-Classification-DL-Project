# Kidney Disease Classification (Deep Learning)

End-to-end deep learning pipeline to classify kidney CT scans into 4 classes: **Cyst, Normal, Stone, Tumor**.  
Built with **TensorFlow**, **Keras**, **ResNet50**, and **MLOps tools** like **DVC** and **MLflow**.

---

## 🚀 Project Overview

- **Problem**: Classify kidney CT scan images for early diagnosis and analysis.
- **Classes**: Cyst | Normal | Stone | Tumor
- **Validation Accuracy**: ~84%

**Key Components**:

- Data ingestion and preprocessing pipeline
- ResNet50 transfer learning model
- Training with Softmax + Categorical Cross-Entropy
- Adam optimizer (LR=0.0001)
- MLOps: DVC for versioned pipelines, MLflow for experiment tracking & model registry
- Evaluation: Confusion matrix and classification report
- Prediction UI: Displays image name, predicted class, and confidence score

---

## 🧠 Improvements vs Original Tutorial

- Modified classifier head for **true 4-class Softmax output**
- Tuned **learning rate** and **Adam optimizer**
- Centralized **hyperparameters in `params.yaml`**
- Added **MLflow model registry & versioning**
- Generated **confusion matrix & classification report**
- Structured project as a **production-ready pipeline** with DVC

---

## 📊 Metrics

**Confusion Matrix**:

[[938 1 170 3]
[ 5 1269 51 198]
[ 12 12 379 10]
[ 136 0 0 548]]

**Classification Report**:

| Class        | Precision | Recall | F1-score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Cyst         | 0.86      | 0.84   | 0.85     | 1112    |
| Normal       | 0.99      | 0.83   | 0.90     | 1523    |
| Stone        | 0.63      | 0.92   | 0.75     | 413     |
| Tumor        | 0.72      | 0.80   | 0.76     | 684     |
| **Accuracy** | -         | -      | 0.84     | 3732    |

---

## 📂 Folder Structure

├── artifacts/ # Training outputs, logs
├── model/ # Trained ResNet50 model (model.h5 tracked with Git LFS)
├── research/ # Original dataset (ignored from GitHub)
├── src/ # Source code (training, evaluation, prediction)
├── config/ # config.yaml, params.yaml
├── dvc.lock # DVC pipeline lock file
├── .gitignore
└── README.md

> ⚠️ Dataset is excluded due to size. Model `model/model.h5` is tracked with **Git LFS**.

---

## ⚡ Installation & Setup

1. **Clone the repository**:

```bash
git clone https://github.com/Siddhant20020/Kidney-Disease-Classification-DL-Project.git
cd Kidney-Disease-Classification-DL-Project
pip install -r requirements.txt
  git lfs install
git lfs pull


🛠 Tech Stack

Deep Learning: TensorFlow, Keras, ResNet50

MLOps: DVC, MLflow

Python Libraries: NumPy, Pandas, OpenCV, Matplotlib

UI: Streamlit / Flask

Version Control: Git, Git LFS


💡 Notes

Dataset is not included (too large)

Model file included via Git LFS

DVC handles pipeline reproducibility

Designed for production-ready DL pipelines
```
