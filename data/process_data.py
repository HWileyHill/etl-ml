import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    #This function takes as arguments the filepaths
    # for two CSV databases, reads them in, merges
    # them, and returns the merged database.
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on='id')
    return df;


def clean_data(df):
    #This function cleans the database df,
    # and returns the cleaned version.
    
    #First, split up the categories column
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.head(1).transpose()[0]
    category_colnames = row.apply(lambda a: a[0:-2])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1:].astype('int')
        
    df = df.drop('categories', axis=1).join(categories)
    
    #Next step: drop duplicate rows
    df = df.drop_duplicates()
    
    #Now get rid of the NaN's and fix the datatypes
    df = df.drop('original', axis=1).dropna()
    for c in df.columns[3:]:
        df[c] = df[c].astype('int')
    
    #And that should do it!
    return df;


def save_data(df, database_filename):
    #This function saves the data to the specified database,
    # returning nothing.
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('YourTableName', engine, index=False, if_exists='replace')
    return;

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()