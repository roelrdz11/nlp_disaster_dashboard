from dash import html
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Define the paths for train dataset
train_path = "C:\\Users\\roelr\\OneDrive\\Documents\\ADAN\\7431\\nlp_disaster_dashboard\\train.csv"

# Load the train dataset
train_data = pd.read_csv(train_path)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Apply preprocessing to the text column
train_data['cleaned_text'] = train_data['text'].apply(preprocess_text)

# Use TF-IDF to convert text into numerical features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(train_data['cleaned_text'])

# Prepare target variable
y = train_data['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
clf_report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Separate disaster and non-disaster tweets for word clouds
disaster_tweets = train_data[train_data['target'] == 1]['cleaned_text']
non_disaster_tweets = train_data[train_data['target'] == 0]['cleaned_text']

# Generate word clouds for both categories
disaster_wordcloud = WordCloud(width=400, height=300, background_color="white").generate(' '.join(disaster_tweets))
non_disaster_wordcloud = WordCloud(width=400, height=300, background_color="white").generate(' '.join(non_disaster_tweets))

# Save the word clouds as images and encode them for embedding in Dash
disaster_buffer = BytesIO()
disaster_wordcloud.to_image().save(disaster_buffer, format="PNG")
disaster_encoded_image = base64.b64encode(disaster_buffer.getvalue()).decode()

non_disaster_buffer = BytesIO()
non_disaster_wordcloud.to_image().save(non_disaster_buffer, format="PNG")
non_disaster_encoded_image = base64.b64encode(non_disaster_buffer.getvalue()).decode()

# Convert confusion matrix to a formatted string for printing
conf_matrix_str = f"""
Confusion Matrix:
    Predicted Non-Disaster | Predicted Disaster
    -------------------------------------------
Actual Non-Disaster |    {conf_matrix[0][0]}             |    {conf_matrix[0][1]}
Actual Disaster     |    {conf_matrix[1][0]}             |    {conf_matrix[1][1]}
"""

# Generate static HTML output
html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLP Disaster Tweet Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .container {{ max-width: 1200px; margin: auto; padding: 20px; }}
        .column {{ float: left; width: 23%; margin: 1%; padding: 10px; border: 1px solid gray; }}
        .clear {{ clear: both; }}
        h1, h3, h4, h5 {{ color: #8a100b; text-align: center; }}
        img {{ width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>NLP Disaster Tweet Dashboard</h1>
        <h3>Roel Rodriguez</h3>
        <h4>Boston College, Woods College: Applied Analytics, ADAN 7431: Natural Language Processing</h4>
        
        <div class="column">
            <h3>Abstract</h3>
            <p>This dashboard presents a Logistic Regression model trained on disaster tweets to predict if a tweet is related to a disaster or not.</p>
        </div>
        
        <div class="column">
            <h3>Methods</h3>
            <p>Preprocessing steps, TF-IDF feature extraction, and Logistic Regression model were used.</p>
        </div>
        
        <div class="column">
            <h3>Results</h3>
            <p>Accuracy: {accuracy:.2f}</p>
            <pre>{conf_matrix_str}</pre>
            <h5>Disaster Tweets Word Cloud</h5>
            <img src="data:image/png;base64,{disaster_encoded_image}">
            <h5>Non-Disaster Tweets Word Cloud</h5>
            <img src="data:image/png;base64,{non_disaster_encoded_image}">
        </div>
        
        <div class="column">
            <h3>Discussion</h3>
            <p>Improvements can be made in reducing false positives.</p>
        </div>
        
        <div class="clear"></div>
    </div>
</body>
</html>
"""

# Save the HTML file
with open("index.html", "w") as file:
    file.write(html_output)
