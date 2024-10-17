import dash
from dash import dcc, html
import pandas as pd
from dash.dependencies import Input, Output
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

# Download NLTK data (if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Define your local paths for train dataset
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

# Initialize the app
app = dash.Dash(__name__)

# Define layout with two word clouds in the Results column
app.layout = html.Div([
    
    # Header Section (Title, Name, Affiliation)
    html.Div([
        html.H1("NLP Disaster Tweet Dashboard"),
        html.H3("Roel Rodriguez"),
        html.H4("Boston College, Woods College: Applied Analytics, ADAN 7431: Natural Language Processing"),
    ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#8a100b'}),
    
    # Main content: 4 columns
    html.Div([
        # Column 1: Abstract, Introduction & Significance
        html.Div([
            html.Div([
                html.H3("Abstract"),
                dcc.Markdown('''
                This dashboard presents a Logistic Regression model trained on disaster tweets to predict if a tweet is related to a disaster or not.
                The goal of this project is to demonstrate the utility of natural language processing (NLP) techniques in identifying relevant
                disaster-related content in social media data, which could assist emergency responders in real-time.
                ''')
            ], style={'overflowY': 'scroll', 'height': '150px', 'padding': '10px'}),
            
            html.Div([
                html.H3("Introduction & Significance"),
                dcc.Markdown('''
                Social media platforms, such as Twitter, generate massive amounts of data during disasters. By leveraging machine learning models,
                it is possible to quickly identify disaster-related content that can be critical for authorities. Accurate identification of this content
                helps in the rapid dissemination of information, early warnings, and coordination during emergency responses.
                ''')
            ], style={'overflowY': 'scroll', 'height': '150px', 'padding': '10px'}),
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'border': '1px solid gray'}),
        
        # Column 2: Methods (without Word Cloud now)
        html.Div([
            html.Div([
                html.H3("Methods"),
                dcc.Markdown('''
                - **Preprocessing**: The text data underwent several preprocessing steps, including:
                    1. Removing URLs, special characters, and punctuation.
                    2. Converting all text to lowercase to ensure uniformity.
                    3. Tokenizing the text into individual words for further analysis.
                    4. Removing stopwords (common words like 'the', 'and', etc.) to focus on key terms.
                    5. Lemmatizing each word, meaning reducing words to their base or root form.
                
                - **Feature Extraction**: Using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer, the text was converted into numerical
                  features. This method highlights the most important words for classification by considering both how frequently they appear in the document
                  and how unique they are across the entire dataset.
                
                - **Model**: A Logistic Regression model was trained to classify each tweet as either related to a disaster or not.
                '''),
            ], style={'overflowY': 'scroll', 'height': '320px', 'padding': '10px'}),
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'border': '1px solid gray'}),
        
        # Column 3: Results (with disaster and non-disaster word clouds)
        html.Div([
            html.Div([
                html.H3("Performance Metrics"),
                html.P(f"Accuracy: {accuracy:.2f}"),
                html.P(f"Precision (Non-Disaster): {clf_report['0']['precision']:.2f}, Recall: {clf_report['0']['recall']:.2f}, F1: {clf_report['0']['f1-score']:.2f}"),
                html.P(f"Precision (Disaster): {clf_report['1']['precision']:.2f}, Recall: {clf_report['1']['recall']:.2f}, F1: {clf_report['1']['f1-score']:.2f}")
            ], style={'overflowY': 'scroll', 'height': '150px', 'padding': '10px'}),
            
            html.Div([
                html.H3("Confusion Matrix"),
                html.Pre(conf_matrix_str)  # Print the confusion matrix as formatted text
            ], style={'padding': '10px'}),

            # Word Clouds
            html.Div([
                html.H3("Disaster Tweets Word Cloud"),
                html.Img(src=f'data:image/png;base64,{disaster_encoded_image}', style={'width': '100%', 'height': 'auto'})
            ], style={'padding': '10px'}),

            html.Div([
                html.H3("Non-Disaster Tweets Word Cloud"),
                html.Img(src=f'data:image/png;base64,{non_disaster_encoded_image}', style={'width': '100%', 'height': 'auto'})
            ], style={'padding': '10px'})
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'border': '1px solid gray'}),
        
        # Column 4: Discussion, Conclusion, and References
        html.Div([
            html.Div([
                html.H3("Discussion"),
                dcc.Markdown('''
                The model achieved a reasonable accuracy and balanced precision and recall for both disaster and non-disaster tweets. 
                However, there are some areas for improvement, particularly in reducing false positives, where non-disaster tweets 
                are mistakenly classified as disaster-related. Future work could explore more sophisticated models or ensemble approaches 
                to improve overall performance and reduce misclassifications.
                ''')
            ], style={'overflowY': 'scroll', 'height': '150px', 'padding': '10px'}),
            
            html.Div([
                html.H3("Conclusion"),
                dcc.Markdown('''
                This project demonstrates the capability of natural language processing and machine learning models in classifying tweets 
                related to disasters. Although the Logistic Regression model provides a simple and efficient solution, more complex models 
                may yield better performance. This dashboard serves as a foundation for future work in real-time disaster detection using 
                social media data.
                ''')
            ], style={'overflowY': 'scroll', 'height': '150px', 'padding': '10px'}),
            
            html.Div([
                html.H3("References"),
                dcc.Markdown('''
                - NLTK: Natural Language Toolkit for text preprocessing.
                - Scikit-learn: Machine learning library for model training and evaluation.
                - TF-IDF: A method for converting text data into numerical features for classification.
                ''')
            ], style={'overflowY': 'scroll', 'height': '150px', 'padding': '10px'}),
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'border': '1px solid gray'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),  # Flexbox layout to arrange columns

])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
