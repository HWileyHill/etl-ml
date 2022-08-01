import sys

#Libraries used in the ML Pipeline file
import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sqlalchemy import create_engine

def load_data(database_filepath):
    # Load in and separate the data.
    # Return X, Y, category_names.
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('YourTableName', engine)
    category_names = df.columns[-36:]
    X = df['message']
    Y = df[category_names]
    return X, Y, category_names;


def tokenize(text):
    # Tokenize the data and return the tokens.
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens;


def build_model():
    #Build and return a machine learning model.
    # Don't train it or anything; later functions will do that.
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(estimator=KNeighborsClassifier()))
])

    return pipeline;


def evaluate_model(model, X_test, Y_test, category_names):
    #Evaluate the model on the test data; print the results.
    # Return nothing.
    Y_pred = pd.DataFrame(data=model.predict(X_test), index=Y_test.index, columns=category_names)

    for c in category_names:
        print('Precision report for ' + c + ' messages:')
        print(classification_report(Y_test[c], Y_pred[c]))
    return;


def save_model(model, model_filepath):
    # Save the model.  Return nothing.
    pickle.dump(model, open(model_filepath, 'wb'))
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()