Fake News Detection using Machine Learning
Overview
This project aims to develop a machine learning-based system to identify and classify news articles as Fake or Real. Given the potential societal harm caused by misinformation, this tool provides an automated, reliable way to evaluate news authenticity. It leverages multiple classifiers to enhance prediction accuracy.
System Highlights
Implements four classification techniques:
Logistic Regression
Decision Tree Classifier
Gradient Boosting Classifier
Random Forest Classifier
Preprocesses textual data using TF-IDF vectorization.
Provides a user-friendly Streamlit web interface for quickfake news classification.
Trains and saves models as pickle files (model.pkl, tfidf.pkl) for deployment.
Supports flexible handling of large model files through options like Git LFS or external hosting.
Dataset
Contains labeled news articles, divided into:
True: Authentic news
Fake: Fabricated or misleading news
Ensure you extract and place Fake.csv and True.csv in your project directory before training.
Dependencies
Install the required Python libraries:
bash
pip install pandas numpy scikit-learn seaborn matplotlib streamlit
Optionally, for large model files, consider using Git Large File Storage (LFS).
Setup & Usage
1. Clone the GitHub repository
bash
git clone https://github.com/yourusername/Fake-News-Detection.git
cd Fake-News-Detection
2. Prepare the dataset
Extract your datasets (e.g., from ZIP or other sources).
Place Fake.csv and True.csv into your project folder.
3. Train the models
To train and save models for deployment:
bash
python model_trainer.py
This script loads datasets, preprocesses data, trains models (Logistic Regression, Decision Tree, Gradient Boosting, Random Forest), and saves them as model.pkl and tfidf.pkl.
4. Run the Streamlit app
For an interactive prediction dashboard:
bash
python -m streamlit run app.py
Note: If streamlit is not recognized, use:
bash
python -m streamlit run app.py
Handling Large Model Files
If your model files (model.pkl, tfidf.pkl) exceed GitHub's size limits, consider:
Using Git Large File Storage (LFS):
Install with: git lfs install
Track files:
bash
git lfs track "*.pkl"
Commit and push as usual.
External hosting:
Store files in cloud storage (e.g., Google Drive, Dropbox).
Download dynamically within app.py at runtime.
Model Training & Deployment Workflow
Train models:
bash
python model_trainer.py
Ensure model files (model.pkl, tfidf.pkl) are in your project directory or accessible via LFS/external storage.
Launch the app:
bash
python -m streamlit run app.py
Project Directory Structure
text
Fake-News-Detection/
│
├── app.py                  # Streamlit app for real-time prediction
├── model_trainer.py        # Script to train ML models
├── requirements.txt        # Required dependencies
├── Fake.csv                # Fake news dataset
├── True.csv                # True news dataset
├── model.pkl               # Trained classifier (or Git LFS/tracked)
└── tfidf.pkl               # TF-IDF vectorizer
Additional Notes
Model Evaluation: Your trained models' performance metrics (accuracy, precision, recall, F1-score) are generated during training and should be included in your project reports.
Model Re-training: Re-run model_trainer.py after updating your datasets to generate new models.
Deployment: The Streamlit app (app.py) reads the saved models to perform on-the-fly classification of input news articles.
Contribution & Customization
Feel free to extend the system:
Add more classifiers
Improve preprocessing
Incorporate multi-language support
Deploy on cloud platforms (Heroku, Streamlit Cloud, etc.)
Summary
This updated README captures:
Your multi-classifier approach
Model training process
Deployment instructions with model file handling strategies
Usage of Streamlit for interactive dashboard
Structure for easy setup and execution
Let me know if you'd like me to generate specific sections, add images, or customize further!
