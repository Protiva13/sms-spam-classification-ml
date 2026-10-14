The project explores different machine learning algorithms, text representation techniques, and class imbalance handling methods to identify the best-performing approach for SMS spam detection.

The final model uses **Character-level TF-IDF features with a Balanced Linear SVM classifier**, achieving the highest performance among all tested approaches.

---

## Dataset

Dataset used:

**SMS Spam Collection Dataset**

The dataset contains SMS messages categorized into two classes:

- `ham` → Normal messages
- `spam` → Spam messages

Dataset source:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## Project Workflow

The complete pipeline consists of:

1. Data loading and exploration
2. Text preprocessing
   - Lowercasing
   - URL removal
   - Email removal
   - Number removal
   - Punctuation removal
   - Stopword removal
   - Lemmatization
3. Train-test splitting
4. TF-IDF feature extraction
5. Machine learning model training
6. Performance evaluation
7. Feature and model improvement experiments

---

## Experiments Performed

### 1. Baseline Model

The original implementation used:

- TF-IDF Vectorization
- Linear SVM classifier

Results:

| Metric | Score |
|---|---:|
| Accuracy | 98.45% |
| Precision | 98.32% |
| Recall | 89.31% |
| F1-score | 93.60% |

---

# Model Comparison

Different classifiers were evaluated using the same TF-IDF representation.

Models tested:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- Linear SVM

### Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 95.74% | 100.00% | 66.41% | 79.82% |
| Naive Bayes | 96.81% | 99.00% | 75.57% | 85.71% |
| Random Forest | 97.49% | 98.17% | 81.68% | 89.17% |
| Linear SVM | 98.45% | 98.32% | 89.31% | 93.60% |

**Observation:**  
Linear SVM provided the strongest balance between precision and recall for SMS spam detection.

---

# TF-IDF Feature Improvements

## Improved Word-level TF-IDF

The original TF-IDF representation was enhanced by:

- Increasing vocabulary size
- Using larger n-grams
- Applying sublinear term frequency scaling

Configuration:

```python
max_features = 15000
ngram_range = (1,3)
sublinear_tf = True

# Author

Tasmim Anan Protiva

GitHub: https://github.com/Protiva13
