import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#load preprocessed dataset
df_reviews = pd.read_csv('data/pre_yelp_500k.csv')

#for the testing purpose, we will only use 100k entries
df_reviews_100k = df_reviews[:100000]

#label positive, negative, neutral sentiments based on the star ratings
condition = [(df_reviews_100k['stars']>=4), (df_reviews_100k['stars']<=2), (df_reviews_100k['stars']==3)]
values = ['positive','negative','neutral']
df_reviews_100k['sentiment']=np.select(condition, values)

#save in CSV
#df_reviews_100k.to_csv('data/pre_yelp_100k_sentiment.csv', index=False)

#checkpoint: load 100k sentiment data from csv
df_reviews_sentiment = pd.read_csv('data/pre_yelp_100k_sentiment.csv')

#plot the sentiment distribution
plt.figure(figsize=(6,6))
df_reviews_sentiment['sentiment'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.title('Review Sentiments');

#we will exclude neutral sentiments as we want to focus on extracting positive/negative sentiments
#for the testing purpose, we will only use 10k entries
df_reviews_sentiment = df_reviews_sentiment[df_reviews_sentiment['sentiment']!='neutral'][:10000]
df_reviews_sentiment


##### TF-IDF - Feature Extraction #####
#TF-IDF using Scikit-Learn library
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#max_features:uses the 2,500 most frequently occuring words - removed
#min_df:include words that occur in at least 7 documents
#max_df:use words that occur in a maximum of 80% of the documents
vectorizer = TfidfVectorizer (min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(df_reviews_sentiment['text'].values.astype('U')).toarray()

#split into training and test datasets in 8:2 ratio
X_train, X_test, y_train, y_test = train_test_split(processed_features, df_reviews_sentiment['sentiment'], test_size=0.2, random_state=0)


######## 1. LOGISTIC REGRESSION ########

# fitting a logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Fitting Logistic regression to the training set
logreg = LogisticRegression(solver='lbfgs',multi_class='auto',random_state=1)
logreg.fit(X_train, y_train)

# Predicting the test set results
logreg_pred = logreg.predict(X_test)

# Training score
print("Score on training set: ", logreg.score(X_train,y_train))
print("Score on test set: ",logreg.score(X_test,y_test))
print(confusion_matrix(y_test,logreg_pred))
print(classification_report(y_test,logreg_pred))
print("accuracy score: ",accuracy_score(y_test, logreg_pred))

"""
[[ 364  151]
 [  26 1459]]
              precision    recall  f1-score   support

    negative       0.93      0.71      0.80       515
    positive       0.91      0.98      0.94      1485

    accuracy                           0.91      2000
   macro avg       0.92      0.84      0.87      2000
weighted avg       0.91      0.91      0.91      2000

0.9115
"""

def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """
    sns.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
 
    plt.title("Confusion Matrix")
 
    sns.set(font_scale=1.4)
    ax = sns.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt='g')
 
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set(ylabel='Actual values', xlabel='Predication')

    #plt.savefig(output_filename, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close()

# define data
data = confusion_matrix(y_test,y_pred_logreg)

# define labels
labels = ["Negative", "Positive"]

# plot confusion matrix
plot_confusion_matrix(data, labels, "confusion_matrix.png")


######## 2. CART CLASSIFIER ########

from sklearn.tree import DecisionTreeClassifier

# Create and train decision tree classifer object
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

# Predicting the test set
clf_pred = clf.predict(X_test)

# training score
print(confusion_matrix(y_test,clf_pred))
print(classification_report(y_test,clf_pred))
print("accuracy score: ",accuracy_score(y_test, clf_pred))

"""
[[ 329  186]
 [ 172 1313]]
              precision    recall  f1-score   support

    negative       0.66      0.64      0.65       515
    positive       0.88      0.88      0.88      1485

    accuracy                           0.82      2000
   macro avg       0.77      0.76      0.76      2000
weighted avg       0.82      0.82      0.82      2000

accuracy score:  0.821
"""

#visualize the tree
import graphviz

dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=vectorizer.get_feature_names(),  
                                filled=True, max_depth=2)
graphviz.Source(dot_data, format="png") 

# define data
data = confusion_matrix(y_test,clf_pred)
 
# plot confusion matrix
plot_confusion_matrix(data, labels, "confusion_matrix.png")


######## 3. RANDOM FOREST ########

#Build a Random Forest classifier model
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

# Predicting the test set
predictions = text_classifier.predict(X_test)

#training score
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print("accuracy score: ",accuracy_score(y_test, predictions))

"""
[[ 312  203]
 [  22 1463]]
              precision    recall  f1-score   support

    negative       0.93      0.61      0.73       515
    positive       0.88      0.99      0.93      1485

    accuracy                           0.89      2000
   macro avg       0.91      0.80      0.83      2000
weighted avg       0.89      0.89      0.88      2000

0.8875
"""

# define data
data = confusion_matrix(y_test,predictions)

# plot confusion matrix
plot_confusion_matrix(data, labels, "confusion_matrix.png")


######## 4. XGBOOST ########

from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error

#Build XGB model
xgb_model = XGBClassifier(random_state=1)
xgb_model.fit(X_train, y_train)

# Predicting the test set
xgb_pred = xgb_model.predict(X_test)


#training score
print("XGBoost score - trainnig set: ",xgb_model.score(X_train, y_train))
print("XGBoost score - test set: ",xgb_model.score(X_test, y_test))
print(confusion_matrix(y_test,xgb_pred))
print(classification_report(y_test,xgb_pred))
print("accuracy score: ",accuracy_score(y_test, xgb_pred))

"""
XGBoost score - trainnig set:  0.985875
XGBoost score - test set:  0.9135
[[ 396  119]
 [  54 1431]]
              precision    recall  f1-score   support

    negative       0.88      0.77      0.82       515
    positive       0.92      0.96      0.94      1485

    accuracy                           0.91      2000
   macro avg       0.90      0.87      0.88      2000
weighted avg       0.91      0.91      0.91      2000

accuracy score:  0.9135
"""

# define data
data = confusion_matrix(y_test,XGB_pred)

# plot confusion matrix
plot_confusion_matrix(data, labels, "confusion_matrix.png")


######## 5. Multilayer Perceptron Classifier ########

from sklearn.neural_network import MLPClassifier

#multilayer perceptron classifier
mlp = MLPClassifier()
mlp.fit(X_train,y_train)

# Predicting the test set
mlp_pred = mlp.predict(X_test)

#training score
print("Confusion Matrix for Multilayer Perceptron Classifier:")
print(confusion_matrix(y_test,mlp_pred))
print("Classification Report:")
print(classification_report(y_test,mlp_pred))
print("accuracy score:",accuracy_score(y_test,mlp_pred))

"""
Confusion Matrix for Multilayer Perceptron Classifier:
[[ 414  101]
 [  62 1423]]
Classification Report:
              precision    recall  f1-score   support

    negative       0.87      0.80      0.84       515
    positive       0.93      0.96      0.95      1485

    accuracy                           0.92      2000
   macro avg       0.90      0.88      0.89      2000
weighted avg       0.92      0.92      0.92      2000

accuracy score: 0.9185
"""

# define data
data = confusion_matrix(y_test,mlp_pred)

# plot confusion matrix
plot_confusion_matrix(data, labels, "confusion_matrix.png")

