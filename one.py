import csv      
import pandas as pd
import re
import numpy as np 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


#train  = pd.read_csv('F:/Datasets/e.csv')
train = pd.read_csv('C:/Users/hasanmova/Desktop/LEMA/tweets.csv', sep=',', usecols=[0,1,2] , names=['id', 'label', 'tweet'] )

print(train['tweet'].head(5))

def remove_pattern(input_txt):
    input_txt = re.sub("@[\w]*", " ", str(input_txt) )
    return input_txt  
    
train['tweet'] = np.vectorize(remove_pattern)(train['tweet'])
train['tweet'] = train['tweet'].str.replace("[^a-zA-Z#]", " ")
train['tweet'] = train['tweet'].apply(lambda x:' '.join(w for w in x.split() if len(w)>3))

tokenized_tweet = train['tweet'].apply(lambda x: word_tokenize(x))

print(tokenized_tweet.head(10))


stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])


tokenized_tweet.to_csv('C:/Users/hasanmova/Desktop/LEMA/st.csv')


