# Resume Screening App
This Resume Screening App is designed to classify resumes into predefined job categories using Natural Language Processing (NLP) techniques. It employs machine learning algorithms to analyze the content of resumes and predict the category the candidate is applying for.

## Features
- Resume Category Classification: Predicts job categories based on the resume content.
- Text Cleaning: The app performs various preprocessing tasks like removing URLs, special characters, and extra spaces from resumes.
- Machine Learning: Uses a K-Nearest Neighbors (KNN) classifier wrapped in a OneVsRest strategy for multi-label classification.
- Interactive Interface: The app is built using Streamlit, providing an easy-to-use interface for uploading resumes and obtaining predictions.
- Data Visualization: Plots the distribution of resumes across different job categories.

## Installation
1. Clone the repository:
git clone <repository-link>
cd resume-screening-app
2. Install required libraries: Ensure you have Python 3.x installed. Then, install the necessary libraries:
pip install -r requirements.txt
3. Download NLTK datasets: The app requires some NLTK datasets. Run the following commands:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
4. Run the Streamlit app: Start the app by running:
streamlit run app.py

## Files and Directories
- app.py: The main Streamlit app file where the UI and prediction logic are implemented.
- clf.pkl: Pre-trained classification model saved as a pickle file.
- tfidf.pkl: Trained TF-IDF vectorizer for text transformation.
- UpdatedResumeDataSet.csv: The dataset containing resumes and their corresponding job categories.
- requirements.txt: Contains all the required libraries and dependencies for the project.

## Dataset
The app uses a dataset from Kaggle consisting of resumes and their respective job categories. The dataset has two columns:
- Category: The job category (e.g., Data Science, Java Developer, etc.).
- Resume: The full content of the resume.

The dataset is cleaned to remove duplicate entries and unnecessary characters using the cleanResume function.

## How It Works
- Text Preprocessing: The cleanResume function is used to clean resume text by removing URLs, special characters, and excess spaces.
- TF-IDF Vectorization: The resumes are transformed into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
- Model Training: The app uses a K-Nearest Neighbors (KNN) classifier for training. The model is trained on the processed resume data and saved as a pickle file.
- Prediction: When a new resume is uploaded, the app cleans the resume, vectorizes it, and uses the pre-trained classifier to predict the job category.

## How to Use
1. Upload a Resume: On the app's interface, upload a resume in plain text format.
2. View Prediction: The app will output the predicted job category based on the resume's content.

## Dependencies
Python 3.x
Streamlit
scikit-learn
NLTK
pandas
seaborn
matplotlib

You can install all the dependencies by running:
pip install -r requirements.txt

## Model Evaluation
The model is evaluated using the following metrics:
- Accuracy: 82.4% on the test set.

## Future Improvements
- Support for more job categories.
- Ability to handle resumes in PDF or Word formats.
- Improve the accuracy of the model by experimenting with different classifiers.
