#!/usr/bin/env python
# coding: utf-8

# **1. Load JSON file**<Br>
# **2. Data Exploration and Visualization**<br>
# **3. Select variables and Convert into CSV**<br>
# **4. Text Preprocessing**
# > a) Change to lower cases<Br>
# > b) Transform links (tentative?)<br>
# > c) Remove punctuation<br>
# > d) Remove stopwords<br>
# > e) lemmatize words (to root forms)<br>

####### 1. Loading JSON file #######
import numpy as np
import pandas as pd
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
get_ipython().magic(u'matplotlib inline')
inline_rc = dict(mpl.rcParams)
from tqdm import tqdm


#True: all data (about 8 mil); False: 500,000 entries
full_data = False

#load user review data
reviews = []
with open('data/yelp_academic_dataset_review.json') as f:
    for i, line in tqdm(enumerate(f)):
        reviews.append(json.loads(line))
        if full_data==False and i+1 >= 500000:
            break

df_review = pd.DataFrame(reviews)
df_review.tail()

#load business data
biz=[]
with open('data/yelp_academic_dataset_business.json') as f1:
    for i, line in tqdm(enumerate(f1)):
        biz.append(json.loads(line))
        if full_data==False and i+1 >= 500000:
            break
df_biz = pd.DataFrame(biz)
df_biz.tail()

#load user data
user=[]
with open('data/yelp_academic_dataset_user.json') as f1:
    for i, line in tqdm(enumerate(f1)):
        user.append(json.loads(line))
        if full_data==False and i+1 >= 500000:
            break
df_user = pd.DataFrame(user)
df_user.tail()


####### 2. Data Exloration and Visualization #######

x=df_review['stars'].value_counts()
x=x.sort_index()

#plot star rating distribution
plt.figure(figsize=(6,5))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Star Rating Distribution", fontsize=16)
plt.ylabel('Number of businesses')
plt.xlabel('Star Ratings')


biz_cat = ''.join(df_biz['categories'].astype('str'))

cats=pd.DataFrame(biz_cat.split(','),columns=['categories'])

#prep for chart
x=cats.categories.value_counts()
x=x.sort_values(ascending=False)
x=x.iloc[0:20]

#chart
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)#,color=color[5])
plt.title("Top business categories",fontsize=25)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)
plt.ylabel('Number of businesses', fontsize=12)
plt.xlabel('Category', fontsize=12)

plt.show()


####### 3. Select Variables and Convert into CSV #######

# Issues for consideration:<br>
# Are we going to pick an industry, then work with the subset businesses? 
# Or we do not consider the industry? e.g. cafe, restaurant, hair salon, etc.

# Replace business_id with businesss name
# Selected three variables: business_name, stars, text
df_comb=df_review.copy()
df_comb['business_name'] = df_comb['business_id'].map(df_biz.set_index('business_id')['name'])
df_comb = df_comb[['business_name','stars','text']]
df_comb


#plot 20 most reviewed business
x=df_comb['business_name'].value_counts()
x=x.sort_values(ascending=False)
x=x.iloc[0:20]

#plot chart
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)#,color=color[5])
plt.title("20 Most Reviewed Businesses",fontsize=20)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)
plt.ylabel('Number of reviews', fontsize=12)
plt.xlabel('Business', fontsize=12)
plt.show()


### Conversion into CSV ###

#Convert review, business, user datasets into CSV
#df_review.to_csv('data/yelp_reviews.csv', index=False)
#df_biz.to_csv('data/yelp_business.csv', index=False)
#df_user.to_csv('data/yelp_user.csv', index=False)


####### 4. Text Preprocessing #######

#### Preprocessing steps: ####
# For Sentiment analysis:
# >a) Change to lower cases
# >b) Remove HTML
# >c) Remove duplicate characters
# >d) Remove punctuation & Tokenize
# >e) Remove stopwords
# >f) Lemmatization/Stemming


import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from tqdm.auto import tqdm, trange
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import string
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

#True: preprocessing for sentiment analysis
#False: preprocessing for text summarization
sentiment=True

def preprocess(s):
    if sentiment==True:
        #1. lowercase
        s = s.lower()
        #2. remove HTML
        soup = BeautifulSoup(s,'lxml')
        html_free = soup.get_text()
        #3. remove duplicate characters
        reg = re.sub(r'([a-z])\1+', r'\1', s)
        #4. Remove punctuation & Tokenize
        no_punct = "".join([c for c in reg if c not in string.punctuation])
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(no_punct)
        #4. Remove stopwords
        filtered_words = [w for w in tokens if w not in stopwords.words('english')]
        #5. lemmatize/stem words
        final_words=[lemmatizer.lemmatize(w) for w in filtered_words]
        #final_words=[stemmer.stem(w) for w in filtered_words]
    else:
        #1. lowercase
        s = s.lower()
        #2. remove HTML
        soup = BeautifulSoup(s,'lxml')
        html_free = soup.get_text()
        #3. remove duplicate characters
        reg = re.sub(r'([a-z])\1+', r'\1', s)
        tokenizer = RegexpTokenizer(r'\w+')
        final_words = tokenizer.tokenize(reg)   
    return " ".join(final_words)

tqdm.pandas()
df_pre['text']=df_pre['text'].progress_map(lambda s:preprocess(s))

#printout before & after of preprocessing
pd.DataFrame({'from': df_review['text'], 'to': df_pre['text']})

#save preprocessed data into CSV
df_pre.to_csv('data/yelp_pre.csv', index=False)

csv_df = pd.read_csv('data/yelp_pre.csv')
#csv_df.index +=1
#csv_df.drop(['Unnamed: 0'],axis=1)
csv_df
