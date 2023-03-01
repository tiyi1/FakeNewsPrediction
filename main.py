import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

# printing the stopwords in English
print(stopwords.words('english'))


# Data Pre-processing
# loading the dataset to a pandas DataFrame

news_dataset = pd.read_csv('train-2.csv')
news_dataset.shape

# print the first 5 rows of the dataframe
news_dataset.head()

# counting the number of missing values in the dataset
news_dataset.isnull().sum()

# replacing the null values with empty string
news_dataset = news_dataset.fillna('')

# merging the author name and news title
news_dataset['content'] = news_dataset['author'] + ' ' +news_dataset['title']

# separating the data & level
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

'''
   Now comes the stemming process (i.e. reducing a word to it's root word). For instance,
    actor, actress, acting --> act'''

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)   # removing all that differs from alphabetic characters
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

#separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

