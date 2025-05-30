# Bee Audio Classification - Queen Presence Detection

This project aims to classify whether a queen bee is present in a hive using audio recordings. It explores both traditional machine learning and deep learning techniques for binary classification of 60-second audio segments.

## 📁 Dataset
Datasets used: [Kaggle dataset bee or not bee](https://www.kaggle.com/datasets/chrisfilo/to-bee-or-no-to-bee)

Git hub reference I used from the publisher itself [GIthub repository link from the publisher on the project used as reference](https://github.com/inesnolas/Audio_based_identification_beehive_states/blob/master/Bee_NotBee_classification/utils.py#L470)

Research Paper I went for [Research Paper](https://zenodo.org/records/1321278/preview/Dataset%20documentation.pdf?include_deleted=0)
- The dataset contains 60-second hive audio recordings.
- Labels are binary: `"bee"` (queen present) and `"nobee"` (queen absent).
- MFCC features (20 coefficients) are precomputed for each audio file.
- Splits are defined in JSON files for train, validation, and test sets.

## 🧠 Models Used

### ✅ Classical ML Model
- **SVM (Support Vector Machine)** using MFCC features (flattened)

### ✅ Deep Learning Models
1. **CRNN (Convolutional Recurrent Neural Network)**  
   - Conv layers for local time-frequency patterns  
   - GRU layer for temporal modeling  
   - Sigmoid for binary classification  

2. **CNN (Convolutional Neural Network)**  
   - Stack of Conv2D layers  
   - Final linear layers for classification  

## 🛠️ Training Setup

- **Loss Function**: `BCEWithLogitsLoss`  
- **Optimizer**: `Adam` with optional weight decay  
- **Scheduler**: `ReduceLROnPlateau` for LR tuning  
- **Early Stopping**: Monitors validation loss with configurable patience  

## 🧪 Evaluation Metrics

Each model is evaluated on the test set using:
- Accuracy
- Precision
- Recall
- F1-Score
- AU-ROC (Area Under Receiver Operating Characteristic)
- AU-PRC (Area Under Precision-Recall Curve)
- Confusion Matrix

## 🔍 Observations

- **SVM** performed decently with MFCC features.
- **CRNN** showed improved performance by capturing temporal patterns.
- **CNN** alone struggled—likely due to lack of recurrent modeling for time sequences.

## 🧾 How to Run

1. Extract MFCC features and set up JSON split files.
2. Choose the model (SVM, CRNN, CNN, CNN+LSTM).
3. Run training via `train_model()` or `train_crnn()` with appropriate datasets.
4. Evaluate using the provided metrics functions.

---

## 🧑‍💻 Contributors
- Aparna Ramesh

## 📄 License
MIT License
