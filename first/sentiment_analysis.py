import pandas as pd
from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load your dataset (change 'sentiment-emotion-labelled_Dell_tweets.csv' to your file's name)
df = pd.read_csv(r"C:\Users\HP\Downloads\sentiment-emotion-labelled_Dell_tweets.csv")

# Assuming 'Text' is the column containing your text data, and 'sentiment' is the column with labels
X = df['Text']
y = df['sentiment']

# Split your data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer with the same parameters used during training
tfidf_vectorizer = TfidfVectorizer(max_features=10000)

# Transform your text data to TF-IDF features for training and testing data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create a logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)

# Train the machine learning model
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model's performance on the test data
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Define a function to predict sentiment for new text
def predict_sentiment(text):
    # Transform the new text into TF-IDF features using the same vectorizer
    new_text_tfidf = tfidf_vectorizer.transform([text])
    
    # Predict the sentiment label
    predicted_sentiment = model.predict(new_text_tfidf)
    
    return predicted_sentiment[0]
#<!-- <a href="{% url 'sentiment_form.html' %}">Analyze Another Text</a>-->
# Example usage for predicting sentiment of new text
new_text = "you are genius guruji great, i will always express my gratitude to you"
predicted_sentiment = predict_sentiment(new_text)
print(f"Predicted Sentiment: {predicted_sentiment}")
