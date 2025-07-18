Fake News Detection Project
A machine learning solution to automatically classify news articles as real or fake, helping combat the spread of misinformation.

🚀 Project Overview
Fake news poses serious risks to society by spreading false information rapidly online. This project trains, evaluates, and deploys a suite of machine learning classifiers to distinguish genuine news from fabricated content. You can reproduce the results, retrain models on new data, and explore each classifier’s performance metrics.

📂 Repository Structure
<img width="1488" height="966" alt="image" src="https://github.com/user-attachments/assets/cacfdf00-5138-47b1-9873-09ac3ea9a5c0" />


🔧 System Requirements
Hardware
≥ 4 GB RAM
Dual-core CPU (e.g., Intel i3 or equivalent)
≥ 500 MB free disk space

Software
Windows, macOS, or Linux
Anaconda (optional) or Python 3.8+

📦 Dependencies
Install via pip:
<img width="1374" height="172" alt="image" src="https://github.com/user-attachments/assets/9f7a6b0f-3da4-4dbb-b991-2cfa1927a169" />

Contents of requirements.txt:

<img width="1300" height="454" alt="image" src="https://github.com/user-attachments/assets/c04f4438-8cb1-4468-ab6c-91ff46714a81" />

🛠️ Setup & Usage
Clone the repository
<img width="1292" height="223" alt="image" src="https://github.com/user-attachments/assets/76426337-a06d-417c-944a-a0d8cfcc4319" />

Prepare data
Ensure data/Fake.csv and data/True.csv are present.

Train models
<img width="1332" height="178" alt="image" src="https://github.com/user-attachments/assets/1b2b57ea-3e79-4e03-bed3-a3f7b873d489" />

This script will:
Load and preprocess data (tokenization, cleaning)
Fit a TF-IDF vectorizer
1.Train four classifiers:
.Logistic Regression
3.Decision Tree
4.Gradient Boosting
5.Random Forest
Evaluate each model (accuracy, precision, recall, F1-score)
Save the best model and TF-IDF vectorizer to models/

Explore results
Open notebooks/EDA_and_Results.ipynb to view performance dashboards, confusion matrices, and detailed metrics.

Run the Streamlit dashboard
<img width="1290" height="174" alt="image" src="https://github.com/user-attachments/assets/97f79751-fffa-43dd-ba4b-99a3cf6a5fad" />

Enter or paste a news article/headline
View live classification and confidence score

If dashboard reports missing model files, run python src/model_trainer.py first.

📊 Evaluation Metrics
<img width="1468" height="392" alt="image" src="https://github.com/user-attachments/assets/4a1c9fb0-a4d3-4453-94fa-308731a36fb9" />
Metrics computed on held-out test set.

🔄 Model Management
Direct GitHub upload: Suitable if model.pkl & tfidf.pkl < 100 MB.
Git LFS: For larger files—track .pkl files via Git LFS.
External storage: Use src/download_models.py to fetch artifacts at runtime.

💡 Best Practices
Re-train regularly with updated news data.
Monitor model drift; evaluate performance on new samples.
Extend the pipeline with additional feature engineering (e.g., named-entity recognition).
