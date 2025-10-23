# 📧 Spam-NotSpam Classification Project

##  Introduction
This project focuses on building a **spam classifier** that can accurately distinguish between **spam** and **ham (non-spam)** text messages. The motivation behind this work is to strengthen spam detection systems using **Machine Learning** and **Deep Learning** approaches.  
The pipeline includes comprehensive preprocessing, multiple model evaluations, and an ensemble technique to achieve high accuracy and robustness across datasets.

---

## Dataset
- **Sources:**
  - UCI SMS Spam Collection Dataset
  - An additional **Kaggle spam dataset** was merged to enhance generalization and improve model stability.
- **Purpose:** Combining multiple datasets ensures the classifier performs well on varied text distributions and spam patterns.

---

## Preprocessing & Feature Engineering
Text cleaning and preparation were key steps before feeding data into ML and DL models.  
The preprocessing pipeline included:
- Removal of **punctuations and numbers** using `re` (Regular Expressions)
- **Stopwords removal** and **stemming** with `nltk`
- **Vectorization approaches:**
  - **For ML models:** Compared **Bag of Words (BoW)** and **TF-IDF** representations → *TF-IDF produced superior accuracy*
  - **For DL models:** Used **Tokenizer**, **Pad Sequences**, and **Embedding layer** for text representation

---

## Models Implemented

### Machine Learning Models
Tested multiple traditional classifiers including:
- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Classifier (SVC)
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost

**Best-performing models** (MultinomialNB, KNN, SVC, Logistic Regression, RandomForest, XGBoost) were combined using a **Soft Voting Classifier**, leading to excellent overall performance.

**Results (ML):**
- **Accuracy:** 0.957
- **Precision:** 0.975

---

### Deep Learning Model
A simple yet effective deep learning architecture was implemented using **LSTM** layers:

`Embedding → LSTM → Dense → Output Layer`

**Results (DL):**
- **Accuracy:** 0.963
- **Precision:** 0.932

---

## Tech Stack
- **Programming Language:** Python
- **Libraries & Tools:**
  - `scikit-learn`, `nltk`, `xgboost`, `tensorflow / keras`, `numpy`, `pandas`, `re`, `matplotlib`
- **Environment:**  Google Colab

---

## Evaluation Metrics
To evaluate performance, the following metrics were used:
- **Accuracy Score**
- **Precision Score**

---

## Conclusion
The project successfully demonstrates the effectiveness of both **classical ML** and **neural network-based** approaches in spam detection.
- The **Voting Classifier** achieved the highest **precision (0.975)**, making it ideal for scenarios where false positives (ham marked as spam) must be minimized.
- The **LSTM model** captured sequential dependencies in text, slightly improving overall accuracy.

This end-to-end approach can be extended to **email filtering**, **social media moderation**, and other **text classification** domains.

---

## Future Enhancements
- Integration with a **web API / streamlit app** for real-time spam detection
- Experimenting with **pretrained embeddings (GloVe / Word2Vec)**
