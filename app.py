from dash import html
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
import base64
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Define the paths for train and test datasets
train_path = "C:\\Users\\roelr\\OneDrive\\Documents\\ADAN\\7431\\nlp_disaster_dashboard\\train.csv"
test_path = "C:\\Users\\roelr\\OneDrive\\Documents\\ADAN\\7431\\nlp_disaster_dashboard\\test.csv"

# Load the train and test datasets
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Preprocessing function for text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Apply preprocessing to the text column for both train and test sets
train_data['cleaned_text'] = train_data['text'].apply(preprocess_text)
test_data['cleaned_text'] = test_data['text'].apply(preprocess_text)

# Handle missing metadata by filling with 'unknown'
train_data['keyword'] = train_data['keyword'].fillna('unknown')
test_data['keyword'] = test_data['keyword'].fillna('unknown')
train_data['location'] = train_data['location'].fillna('unknown')
test_data['location'] = test_data['location'].fillna('unknown')

# Encode categorical metadata (e.g., 'keyword' and 'location')
encoder = LabelEncoder()
train_data['keyword_encoded'] = encoder.fit_transform(train_data['keyword'])

# Add a generic 'unknown' label to handle unseen test data values
unknown_value = 'unknown'
if unknown_value not in encoder.classes_:
    encoder.classes_ = np.append(encoder.classes_, unknown_value)

# Transform the test data, mapping unseen values to 'unknown'
test_data['keyword_encoded'] = test_data['keyword'].apply(lambda x: x if x in encoder.classes_ else unknown_value)
test_data['keyword_encoded'] = encoder.transform(test_data['keyword_encoded'])

# Same procedure for 'location'
train_data['location_encoded'] = encoder.fit_transform(train_data['location'])
if unknown_value not in encoder.classes_:
    encoder.classes_ = np.append(encoder.classes_, unknown_value)

test_data['location_encoded'] = test_data['location'].apply(lambda x: x if x in encoder.classes_ else unknown_value)
test_data['location_encoded'] = encoder.transform(test_data['location_encoded'])

# Use TF-IDF to convert text into numerical features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_text_train = tfidf_vectorizer.fit_transform(train_data['cleaned_text'])
X_text_test = tfidf_vectorizer.transform(test_data['cleaned_text'])

# Combine text features with metadata (keyword and location)
X_train = pd.DataFrame(X_text_train.toarray())
X_train['keyword'] = train_data['keyword_encoded']
X_train['location'] = train_data['location_encoded']

X_test = pd.DataFrame(X_text_test.toarray())
X_test['keyword'] = test_data['keyword_encoded']
X_test['location'] = test_data['location_encoded']

# Ensure all column names are strings
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Prepare target variable
y_train = train_data['target']

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions for the test set for Kaggle submission
y_pred = model.predict(X_test)

# Save the predictions for submission
submission = pd.DataFrame({'id': test_data['id'], 'target': y_pred})
submission.to_csv("submission.csv", index=False)


# Evaluate the model on the training data
y_train_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, y_train_pred)
clf_report = classification_report(y_train, y_train_pred, output_dict=True)
conf_matrix = confusion_matrix(y_train, y_train_pred)

# Separate disaster and non-disaster tweets for word clouds
disaster_tweets = train_data[train_data['target'] == 1]['cleaned_text']
non_disaster_tweets = train_data[train_data['target'] == 0]['cleaned_text']

# Generate word clouds for both categories
disaster_wordcloud = WordCloud(width=400, height=300, background_color="white").generate(' '.join(disaster_tweets))
non_disaster_wordcloud = WordCloud(width=400, height=300, background_color="white").generate(' '.join(non_disaster_tweets))

# Save the word clouds as images and encode them for embedding in HTML
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

kaggle_score_image_path = r"C:\Users\roelr\OneDrive\Pictures\kaggle_nlp_disaster.png"

# Encode Kaggle score image as base64
with open(kaggle_score_image_path, "rb") as image_file:
    kaggle_score_encoded_image = base64.b64encode(image_file.read()).decode()

# Generate static HTML output with matching structure and layout
html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLP Disaster Tweet Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #b29d6c;  /* Set background color of the page */
            margin: 0;
            padding: 0;
        }}
        .container {{
            max-width: 1200px;
            margin: auto;
            padding: 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;  /* Ensure the columns are spaced out evenly */
        }}
        .column {{
            flex-basis: 23%;  /* Each column takes up about 23% of the width */
            margin: 1%;  /* Small margin between columns */
            padding: 10px;
            border: 1px solid gray;
            box-sizing: border-box;
        }}
        .header {{
            background-color: #8a100b;
            color: white;
            padding: 10px;
            text-align: center;
            width: 100%;
        }}
        h1, h3, h4, h5 {{
            color: white;  /* Ensure header text is visible */
            margin: 10px 0;
        }}
        .section-title {{
            text-align: center;
            font-weight: bold;
        }}
        img {{
            width: 100%;
            height: auto;
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: monospace;  /* Use monospace font for better alignment */
            background-color: #f5f5f5; /* Light background for easier reading */
            padding: 10px;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header Section -->
        <div class="header">
            <h1>NLP Disaster Tweet Dashboard</h1>
            <h3>Roel Rodriguez</h3>
            <h4>Boston College, Woods College: Applied Analytics, ADAN 7431: Natural Language Processing</h4>
        </div>

        <!-- Column 1: Abstract, Introduction & Significance -->
        <div class="column">
            <h3 class="section-title">Abstract</h3>
            <p>This dashboard presents a Logistic Regression model trained on disaster tweets to predict if a tweet is related to a disaster or not. The goal of this project is to demonstrate the utility of natural language processing (NLP) techniques in identifying relevant disaster-related content in social media data, which could assist emergency responders in real-time.</p>
            
            <h3 class="section-title">Introduction & Significance</h3>
            <p>Social media platforms, such as Twitter, generate massive amounts of data during disasters. By leveraging machine learning models, it is possible to quickly identify disaster-related content that can be critical for authorities. Accurate identification of this content helps in the rapid dissemination of information, early warnings, and coordination during emergency responses.</p>
        </div>

        <!-- Column 2: Methods -->
        <div class="column">
            <h3 class="section-title">Methods</h3>
            <p><strong>Preprocessing:</strong> The text data underwent several preprocessing steps, including:</p>
            <ul>
                <li>Removing URLs, special characters, and punctuation.</li>
                <li>Converting all text to lowercase to ensure uniformity.</li>
                <li>Tokenizing the text into individual words for further analysis.</li>
                <li>Removing stopwords (common words like 'the', 'and', etc.) to focus on key terms.</li>
                <li>Lemmatizing each word, reducing words to their base or root form.</li>
            </ul>
            <p><strong>Metadata Handling:</strong> Missing keywords and locations were filled with 'unknown'. Both the keyword and location fields were encoded and added to the model's feature set.</p>
        </div>

        <!-- Column 3: Results -->
        <div class="column">
            <h3 class="section-title">Results</h3>
            <p><strong>Accuracy:</strong> {accuracy:.2f}</p>
            <pre>
Confusion Matrix:
     Predicted Non-Disaster | Predicted Disaster
     ----------------
Actual Non-Disaster  |        4116       |         226
Actual Disaster      |         752       |        2519
            </pre>

            <!-- Word Clouds -->
            <h5>Disaster Tweets Word Cloud</h5>
            <img src="data:image/png;base64,{disaster_encoded_image}">

            <h5>Non-Disaster Tweets Word Cloud</h5>
            <img src="data:image/png;base64,{non_disaster_encoded_image}">
        </div>

        <!-- Column 4: Kaggle Submission, Discussion, and Conclusion -->
        <div class="column">
            <h3 class="section-title">Kaggle Submission</h3>
            <p>See the Kaggle submission score below based on this model.</p>
            <img src="data:image/png;base64,{kaggle_score_encoded_image}" alt="Kaggle Submission Score">

            <!-- Discussion and Conclusion (below the Kaggle score) -->
            <h3 class="section-title">Discussion</h3>
            <p>The model achieved reasonable accuracy and balanced precision and recall for both disaster and non-disaster tweets. However, improvements can be made, particularly in reducing false positives, where non-disaster tweets are mistakenly classified as disaster-related. Future work could explore more sophisticated models or ensemble approaches to improve overall performance.</p>

            <h3 class="section-title">Conclusion</h3>
            <p>This project demonstrates the capability of natural language processing and machine learning models in classifying tweets related to disasters. Although the Logistic Regression model provides a simple and efficient solution, more complex models may yield better performance.</p>

            <h3 class="section-title">References</h3>
            <ul>
                <li>NLTK: Natural Language Toolkit for text preprocessing.</li>
                <li>Scikit-learn: Machine learning library for model training and evaluation.</li>
                <li>TF-IDF: A method for converting text data into numerical features for classification.</li>
            </ul>
        </div>

        <!-- Clear floats -->
        <div class="clear"></div>
    </div>
</body>
</html>
"""

# Save the HTML file
with open("index.html", "w") as file:
    file.write(html_output)
