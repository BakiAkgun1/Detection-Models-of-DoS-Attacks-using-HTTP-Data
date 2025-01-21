
# Detection of DoS (Denial of Service) Attacks Using HTTP Data

This repository contains the code and resources for detecting DoS attacks using HTTP traffic data. The project explores multiple approaches, including traditional machine learning models, RNN-based deep learning models, and a complex neural network architecture.

## Table of Contents

- [Introduction](#introduction)
- [Data Analysis and Feature Engineering](#data-analysis-and-feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Models](#deep-learning-models)
- [Ensemble Neural Network Architecture](#ensemble-neural-network-architecture)
- [Results and Insights](#results-and-insights)
- [References](#references)

## Introduction

The goal of this project is to develop effective methods for detecting DoS attacks by analyzing HTTP data. It includes the following key steps:
1. Exploratory Data Analysis (EDA) and feature engineering.
2. Training and evaluation of traditional and deep learning models.
3. Implementing a state-of-the-art ensemble neural network.

## Data Analysis and Feature Engineering

### Exploratory Data Analysis (EDA)
- **Class Distribution**: 
  - Normal: 1,700 instances
  - Anomalous (DoS): 50,766 instances
    
![image](https://github.com/user-attachments/assets/687c1127-4908-4603-8540-ea929194a9b7)

- **Null Values**: 
  - Handled missing data in `Flow_Byts/s` column (3 instances).
- **Feature Correlation**:
  - Key features: `Dst_Port`, `Src_Port`, `PSH_Flag_Cnt`, and `Flow_IAT_Std`.
![image](https://github.com/user-attachments/assets/685b9554-9368-4931-ad06-f2a8fa0472da)

### Feature Engineering
- Transformed timestamp into `hour`, `minute`, and `day_of_week`.
- Engineered packet length variations and flow distributions.
- Applied PCA for dimensionality reduction.
- Converted categorical columns to numerical format using label encoding.
![image](https://github.com/user-attachments/assets/109a9f9f-c8e3-4e05-854b-6403ba8e83f5)

### Feature Selection
- **Correlation-based**: Retained highly correlated features (`Dst_Port`, `Src_Port`, etc.).
![image](https://github.com/user-attachments/assets/2ff06a53-4ab6-412d-999a-ec671b018ba7)

- **Mutual Information**: Selected features with high dependency on the target.
![image](https://github.com/user-attachments/assets/d7aba5e4-5563-47d9-8aad-77e40804970c)

### Visualizations
![image](https://github.com/user-attachments/assets/11223811-dfbe-4d64-bc2e-0392bfe11f0f)
![image](https://github.com/user-attachments/assets/8615f465-5abf-45d5-b740-cc6c6d90ec52)
![image](https://github.com/user-attachments/assets/87032958-3b01-4d3e-b7f5-636cf5d45916)
![image](https://github.com/user-attachments/assets/bc500dba-4e8d-4dc9-8fcf-c653c1b4167a)


## Machine Learning Models

Implemented traditional classifiers:
1. Random Forest
2. XGBoost
3. LightGBM
4. Logistic Regression

### Evaluation Metrics
![image](https://github.com/user-attachments/assets/5b0b14f6-5075-4329-ba23-ecb1c994de96)

## Deep Learning Models

Explored RNN-based architectures:
1. RNN
2. LSTM
3. GRU

![image](https://github.com/user-attachments/assets/46107acd-7530-4e89-b495-9f64170998f5)

### Performance Metrics
![image](https://github.com/user-attachments/assets/9994f1c8-85b5-40a5-93f1-750764ca78e5)
![image](https://github.com/user-attachments/assets/dbfd03dd-5a37-4b7e-9452-7db485ed250e)


## Ensemble Neural Network Architecture

Developed a hybrid CNN+LSTM+BiLSTM+GRU model:
- **Architecture**:
  - 1D Convolutional Layer
  - LSTM and BiLSTM layers
  - GRU layer
  - Dense output layer
- **Results**: Achieved the highest accuracy, precision, and recall among all models.
![image](https://github.com/user-attachments/assets/8ce89f4a-6832-41f1-ad69-77034a1796af)
![image](https://github.com/user-attachments/assets/f403ee43-9aa3-4194-a005-b27d680e1176)

## Results and Insights

- The **CNN+LSTM+BiLSTM+GRU** ensemble model demonstrated superior performance.
- Deep learning models excel at learning temporal data dependencies.
- Ensemble approaches significantly enhance classification accuracy compared to individual models.

## References

- [Dataset Source](https://www.kaggle.com/datasets/razasiddique/dos-attack-http-dataset)
- [Implementation Reference](https://www.kaggle.com/code/danielagudeydoe/dos-detection-real-time-data-bilstm)
- [Additional Study](https://dergipark.org.tr/en/download/article-file/1371423)
