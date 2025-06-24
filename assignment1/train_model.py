import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Use pandas to load the IMDB Dataset.csv file.
df = pd.read_csv('IMDB Dataset.csv')

# You will need to split your data into features (the review text) and labels (the sentiment). Let's call them X and y.
X = df["review"]
y = df["sentiment"]


# Create a pipeline that first transforms the text data using TfidfVectorizer and then feeds it to the MultinomialNB classifier.
nb_clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words="english")), 
    ('clf', MultinomialNB())
])

# Train the pipeline on your entire dataset (X and y). No need to create a train-test split for this assignment
nb_clf.fit(X,y)

# Use the joblib library to dump your trained Pipeline object into a file named sentiment_model.pkl.
joblib.dump(nb_clf, 'sentiment_model.pkl')