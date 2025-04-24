# Relevance and Emotion Recognition Classification Project

This project focuses on two distinct machine learning tasks: **Relevance Classification** and **Emotion Recognition**, using Python, TensorFlow, Keras, PyTorch, and Scikit-learn.

## Project Overview

### Part 1: Relevance Classification
- **Objective**: Predict the relevance of news articles to users' information needs, assigning a binary judgment (0 = not relevant, 1 = relevant).
- **Dataset**: Articles with features like document ID, author, body, byline, title, and user information requirements (topic ID, description, narrative, topic title). Training data has 21,611 rows.
- **Approach**:
  - Data pre-processing: tokenization, TF-IDF vectorization, sequence generation.
  - Models: Logistic Regression, 3-layer Feed-Forward Neural Network (MLP), LSTM, and BERT.
  - Achieved a Kaggle score of 0.8727 using LSTM, outperforming the baseline MLP (score: 0.83).
- **Key Achievements**:
  - Reduced pre-processing time by 30% through optimized techniques.
  - Improved data processing workflows for cleaner model inputs.
  - Enhanced emotional recognition in text classification.

### Part 2: Emotion Recognition
- **Objective**: Develop a deep learning model to detect and classify emotions from nonverbal cues (facial expressions), with potential to extend to body language and tone of voice.
- **Dataset**: Image data with columns: id, emotion (integer label), pixels (pixel values).
- **Approach**:
  - Explored RandomForestClassifier (Kaggle score: 0.44), CNN (score: 0.23), and ResNet (score: 0.59).
  - Used convolutional neural networks (CNNs) and ResNet for feature extraction.
- **Key Achievements**:
  - Deployed a state-of-the-art emotion recognition system using ResNet, achieving the highest Kaggle score of 0.59.
  - Advanced AI capabilities with complex neural networks for emotion recognition.

## Technologies Used
- Python
- TensorFlow
- Keras
- PyTorch
- Scikit-learn
- pandas, numpy, matplotlib, seaborn

## Project Structure
```
Relevance_Emotion_Classification/
├── datasets/
│   ├── relevance/              # Relevance classification datasets
│   │   ├── relevance_train.csv
│   │   ├── relevance_test.csv
│   │   └── README.md
│   └── emotion/                # Emotion recognition dataset
│       ├── my_emotion_train.csv
│       └── README.md
├── notebooks/
│   ├── relevance_classification.ipynb
│   └── emotion_recognition.ipynb
├── README.md                   # Project overview
├── requirements.txt            # Python dependencies
└── .gitignore                  # Ignored files
```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Relevance_Emotion_Classification.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebooks:
   ```bash
   jupyter notebook notebooks/
   ```

## Datasets
- **Relevance Classification**: Located in `datasets/relevance/`.
  - `relevance_train.csv`: Training dataset with features: doc_id, judgement (0 = not relevant, 1 = relevant), author, body, byline, title, topic_id, description, narrative, topic_title.
  - `relevance_test.csv`: Test dataset with similar features, excluding the judgement column.
- **Emotion Recognition**: Located in `datasets/emotion/`.
  - `my_emotion_train.csv`: Training dataset with columns: id, emotion (integer label), pixels (space-separated pixel values for images).
- If datasets are too large for GitHub, download them from [Google Drive Link] (update with your link if applicable).

## Usage
- `relevance_classification.ipynb`: Contains code and analysis for the relevance classification task.
- `emotion_recognition.ipynb`: Contains code and analysis for the emotion recognition task.
- Follow each notebook to reproduce results or modify for your own datasets.

## Results
- **Relevance Classification**: LSTM model achieved a Kaggle score of 0.8727.
- **Emotion Recognition**: ResNet model achieved a Kaggle score of 0.59.

## Limitations and Future Work
- **Relevance Classification**: Limited by computational resources (Google Colab RAM/GPU constraints). Using a larger dataset could improve accuracy by ±10%.
- **Emotion Recognition**: CNN underperformed (score: 0.23); ResNet was more effective. Future work could involve larger datasets and incorporating body language and tone of voice.

## License
MIT License