import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import spacy
import re
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.utils import resample
from sklearn.naive_bayes import MultinomialNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib

# General Steps to Modelling
# 1- Convert the ratings into Sentiment (Positive or 1 if rating >= 5 else Negative or 0)
# 2- Add together all the comments as not all columns have data in them (Also correlation between the sentiment being expressed in benefits/side effect or the general comment)
# 3- Compare Accuracy from training dataset as well as from test data set accuracy to see how the model performs
# 4- Start with basic models using count vectorizer and tfidf vectorizers but optimize them based on max features, stop words, regex and tokenization
# 5- Use different machine learning models using GridSearchCv to find optimal
# 6- Test the models, if unsatisfactory use a small base case BERT model to see if it can do better sentiment analysis
# 7- Translate it into an interactive streamlit application

# Start of Code
# Loading the dataset into a function to refresh it later for further modelling

def refresh_data():
    # Read Model and out of sample dataset
    df = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    # Converting > 5 rating into Positive (1) else negative (0)
    df['LABEL_COLUMN'] = df.rating.apply(lambda x : 1 if x >=5 else 0)
    # Looking through the data, there are some reviews which have review in 1 column while rest are empty so better idea is to merge them together for analysis
    df['DATA_COLUMN'] = df['benefits_review'] + "" + df['side_effects_review'] + "" +df ['comments_review']
    # Doing the same for the df_test dataset
    df_test['LABEL_COLUMN'] = df_test.rating.apply(lambda x : 1 if x >=5 else 0)
    df_test['DATA_COLUMN'] = df_test['benefits_review'] + "" + df_test['side_effects_review'] + "" +df_test['comments_review']
    # X and y features of training dataset
    X = df.loc[:,'DATA_COLUMN']
    y = df.LABEL_COLUMN
    # Test & Training datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=123)
    train, test = train_test_split(df, test_size=0.3, random_state=123,stratify = y)
    # Converting DF into list for analysis
    X_train_docs = list(X_train)
    X_test_docs =  list(X_test)
    df_test_docs = list(df_test.DATA_COLUMN)
    # return values    
    return df,train,test,df_test,X_train_docs, y_train, X_test_docs, y_test, df_test_docs

# Refreshing and reassigning variables
df,train,test,df_test,X_train_docs, y_train, X_test_docs, y_test,df_test_docs = refresh_data()

# Model 1- Using Count Vectorizer without any text cleaning - 
# 1- Training Dataset Accuracy = 75.91% & F1-Score = 84.27%
# 2- Testing Dataset Accuracy = 75.68% & F1-Score = 83.99%

# Tokenizing the list
vect = CountVectorizer(ngram_range=(1, 3), stop_words="english",max_features=1000).fit(X_train_docs)
X_train_features = vect.transform(X_train_docs)
feature_names = vect.get_feature_names_out()

# Checking output
print("First 100 features:\n{}".format(feature_names[:100])) 
# A lot of these features are numbers -- Should be looked into for following models

# Modelling
lin_svc = LinearSVC(max_iter=120000)
scores = cross_val_score(lin_svc, X_train_features, y_train, cv=5)

# Printing result
print("Mean cross-validation accuracy: {:.3f}".format(np.mean(scores)))

# Modelling & Training Dataset Accuracy/F1 Score
lin_svc.fit(X_train_features, y_train)
X_test_features = vect.transform(X_test_docs)
y_test_pred = lin_svc.predict(X_test_features)
accuracy_M1 = metrics.accuracy_score(y_test, y_test_pred)
f1_score_M1 = f1_score(y_test, y_test_pred)
print(f"The training dataset accuracy for Model 1 is: {accuracy_M1:.2%} and f1 score is {f1_score_M1:.2%}")

# Accuracy and f1 score from Testing Set
df_test_features= vect.transform(df_test_docs)
test_pred = lin_svc.predict(df_test_features)
accuracy_M1_T = metrics.accuracy_score(df_test.LABEL_COLUMN,test_pred)
f1_score_M1_T = f1_score(df_test.LABEL_COLUMN,test_pred)
print(f"The testing dataset accuracy for Model 1 is: {accuracy_M1_T:.2%} and f1 score is {f1_score_M1_T:.2%}")

# Model 2 - Improving model - Using Lemmitization 
# 1- Training Dataset Accuracy = 79.35% & F1-Score = 87.30%
# 2- Test Dataset Accuracy = 80.89% & F1-Score = 88.06%

# Refreshing and reassigning variables
df,train,test,df_test,X_train_docs, y_train, X_test_docs, y_test,df_test_docs = refresh_data()

en_nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

pattern = re.compile('(?u)\\b\\w\\w+\\b')

def custom_tokenizer(document):
    doc_spacy = en_nlp(document)
    lemmas = [token.lemma_ for token in doc_spacy]
    return [token for token in lemmas if token not in STOP_WORDS and pattern.match(token)]

vect = TfidfVectorizer(tokenizer=custom_tokenizer, ngram_range=(1, 2),max_features=1000).fit(X_train_docs)

X_train_features = vect.transform(X_train_docs)

feature_names = vect.get_feature_names_out()
print("First 100 features:\n{}".format(feature_names[:100])) 

# Problem of digits stays
lin_svc = LinearSVC(max_iter=120000)

scores = cross_val_score(lin_svc, X_train_features, y_train, cv=5)
print("Mean cross-validation accuracy: {:.3f}".format(np.mean(scores)))

lin_svc.fit(X_train_features, y_train)

X_test_features = vect.transform(X_test_docs)

y_test_pred = lin_svc.predict(X_test_features)

accuracy_M2 = metrics.accuracy_score(y_test, y_test_pred)
f1_score_M2 = f1_score(y_test, y_test_pred)
print(f"The accuracy for Model 2 is: {accuracy_M2:.2%} and f1 score is {f1_score_M2:.2%}")

# Accuracy and f1 score from testing dataset
df_test_features= vect.transform(df_test_docs)
test_pred = lin_svc.predict(df_test_features)
accuracy_M2_T = metrics.accuracy_score(df_test.LABEL_COLUMN,test_pred)
f1_score_M2_T = f1_score(df_test.LABEL_COLUMN,test_pred)
print(f"The testing dataset accuracy for Model 2 is: {accuracy_M2_T:.2%} and f1 score is {f1_score_M2_T:.2%}")

# Model 3 - Optimizing based on previous results
 
# 1- Training Accuracy = 82.04% & F1-Score = 89.44%
# 2- Testing Accuracy = 81.18% & F1-Score = 88.68%

# Learning from the last 2 models we will see if reworking the following increases accuracy:
# 1- Pattern Matching - Remove digits from the tokens chosen
# 2- Increase token length to 2 and above to remove useless tokens like I, U etc.
# 3- Use GridSearchCV with some machine learning models usually used for NLP to find the best one (with some sample parameters chosen from google searches for starters)
# 4- Use GridSearchCV to optimize number of features chosen as well as ngrams chosen

# Refreshing and reassigning variables
df,train,test,df_test,X_train_docs, y_train, X_test_docs, y_test,df_test_docs = refresh_data()

# Use Spacy's Medium dictionary
en_nlp = spacy.load("en_core_web_md")

# Remove the digits from the features
pattern = re.compile(r'(?u)\b(?![\d_])\w+\b') 

# Remove stop words, digits and those with not a lot of explanatory power not removed by stopwords (I, u etc)
def custom_tokenizer(document):
    doc_spacy = en_nlp(document)
    lemmas = [token.lemma_ for token in doc_spacy]
    return [token for token in lemmas if token not in STOP_WORDS and pattern.match(token) 
            and len(token) >= 2]

vect = TfidfVectorizer(tokenizer=custom_tokenizer, ngram_range=(1, 2), max_features=1000)

feature_names = vect.fit(X_train_docs).get_feature_names_out()

# Checking output
print("First 100 features:\n{}".format(feature_names[:100])) 

# Setting up Class for checkign different estimators

class DummyEstimator(BaseEstimator):
    def fit(self): pass
    def score(self): pass
        
# Create a pipeline
pipe = Pipeline([('clf', DummyEstimator())]) # Placeholder Estimator
    
# Candidate learning algorithms and their hyperparameters
search_space = [
    {
        'clf': [LogisticRegression()],
        'clf__penalty': ['none', 'l2'],
        'clf__C': np.logspace(0, 4, 5)
    },
    {
        'clf': [MultinomialNB()],
        'clf__alpha': np.linspace(0, 1, 5)
    },
    {
        'clf': [LinearSVC()],
        'clf__C': np.logspace(-3, 3, 5)
    },
    {
        'clf': [XGBClassifier()],
        'clf__max_depth': [5,7],
        'clf__learning_rate': [0.01, 0.1,1]
    }
]

# Create grid search 
gs = GridSearchCV(pipe, search_space)
gs.fit(vect.fit_transform(X_train_docs), y_train)

gs.best_estimator_
# Pipeline(steps=[('clf', LinearSVC())])
gs.best_params_
# {'clf': LinearSVC(), 'clf__C': 1.0}

# Check if the accuracy increases. If it does then try to hyperparameter SVC more
y_pred = gs.best_estimator_.predict(vect.transform(X_test_docs))

X_test_features = vect.transform(X_test_docs)
y_test_pred = gs.best_estimator_.predict(X_test_features)
accuracy_M3 = metrics.accuracy_score(y_test, y_test_pred)
f1_score_M3 = f1_score(y_test, y_test_pred)
print(f"The accuracy for Model 1 is: {accuracy_M3:.2%} and f1 score is {f1_score_M3:.2%}")
# So far the accuracy has improved to 80% which means we are on the right track

# Going to take the the linearsvc model and try to improve it further 

# Refreshing and reassigning variables
df,train,test,df_test,X_train_docs, y_train, X_test_docs, y_test,df_test_docs = refresh_data()

def custom_tokenizer(document):
    doc_spacy = en_nlp(document)
    lemmas = [token.lemma_ for token in doc_spacy]
    return [token for token in lemmas if token not in STOP_WORDS and pattern.match(token) 
            and len(token) >= 2]

vect = TfidfVectorizer(tokenizer=custom_tokenizer, ngram_range=(1, 2), max_features=1000)

# SVC search grid
SVCparam_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'loss': ['hinge', 'squared_hinge'],
    'penalty': ['l1', 'l2'],
    'dual': [True, False],
    'tol': [1e-4, 1e-3, 1e-2]
}

grid = GridSearchCV(estimator = LinearSVC() , param_grid = SVCparam_grid, cv=3)

grid.fit(vect.fit_transform(X_train_docs), y_train)
print(f"Best parameters are: {grid.best_params_}")

# Best parameters are: {'C': 1, 'dual': True, 'loss': 'hinge', 'penalty': 'l2', 'tol': 0.0001}

# Optimize TfidfVectorizer

df,train,test,df_test,X_train_docs, y_train, X_test_docs, y_test,df_test_docs = refresh_data()

pipe_tfidf = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=custom_tokenizer)),
    ('clf',LinearSVC(C=1, loss='hinge', penalty='l2', dual=True, tol=0.0001))
    ])

 # Define a search space for the TfidfVectorizer hyperparameters
search_space_tfidf = {
     'tfidf__ngram_range': [(1, 2),(1,3)],
     'tfidf__max_features': [600,850,1000,1500]
 }

# Create grid search for the TfidfVectorizer hyperparameters
gs_tfidf = GridSearchCV(pipe_tfidf, search_space_tfidf)

# Fit grid search to training data
gs_tfidf.fit(X_train_docs, y_train)

# Print the best hyperparameters for the TfidfVectorizer
print("Best TfidfVectorizer parameters:", gs_tfidf.best_params_)

# Best TfidfVectorizer parameters: {'tfidf__max_features': 1000, 'tfidf__ngram_range': (1, 3)}

# Incorporating everything into a final Model 4
df,train,test,df_test,X_train_docs, y_train, X_test_docs, y_test,df_test_docs = refresh_data()

# Use Spacy's Medium dictionary
en_nlp = spacy.load("en_core_web_md")

# Remove the digits from the features
pattern = re.compile(r'(?u)\b(?![\d_])\w+\b') 

# Remove stop words, digits and those with not a lot of explanatory power not removed by stopwords (I, u etc)
def custom_tokenizer(document):
    doc_spacy = en_nlp(document)
    lemmas = [token.lemma_ for token in doc_spacy]
    return [token for token in lemmas if token not in STOP_WORDS and pattern.match(token) 
            and len(token) >= 2]

vect = TfidfVectorizer(tokenizer=custom_tokenizer, ngram_range=(1, 3), max_features=1000)

feature_names = vect.fit(X_train_docs).get_feature_names_out()

# Checking output
print("Features:\n{}".format(feature_names)) 

best_svc_model = LinearSVC(C=1, loss='hinge', penalty='l2', dual=True, tol=0.0001)

best_svc_model.fit(vect.fit_transform(X_train_docs), y_train)
X_test_features = vect.transform(X_test_docs)

# Save the trained model to a file
# joblib.dump(best_svc_model, 'Model3_model.pkl')

scores = cross_val_score(best_svc_model, X_train_features, y_train, cv=5)
print("Mean cross-validation accuracy: {:.3f}".format(np.mean(scores)))

y_test_pred = best_svc_model.predict(X_test_features)
accuracy_M3 = metrics.accuracy_score(y_test, y_test_pred)
f1_score_M3 = f1_score(y_test, y_test_pred)
print(f"The accuracy for Model 3 is: {accuracy_M3:.2%} and f1 score is {f1_score_M3:.2%}")

# Accuracy and f1 score from Testing dataset
df_test_features= vect.transform(df_test_docs)
test_pred = best_svc_model.predict(df_test_features)
accuracy_M3_T = metrics.accuracy_score(df_test.LABEL_COLUMN,test_pred)
f1_score_M3_T = f1_score(df_test.LABEL_COLUMN,test_pred)
print(f"The testing dataset accuracy for Model 3 is: {accuracy_M3_T:.2%} and f1 score is {f1_score_M3_T:.2%}")

# Before making a model 4, lets test if oversampling or undersampling improve Model 3, if they do lets create model 4 otherwise use model 3

# Check model with oversampling undersampling

df,train,test,df_test,X_train_docs, y_train, X_test_docs, y_test,df_test_docs = refresh_data()

df_0 = df.loc[df['LABEL_COLUMN'] == 0,:]
df_1 = df.loc[df['LABEL_COLUMN'] == 1,:]
print(df_0.LABEL_COLUMN.value_counts())
print(df_1.LABEL_COLUMN.value_counts())
df_new= resample(df_1,replace=False,random_state = 123,n_samples = df_0.shape[0])
df_new.LABEL_COLUMN.value_counts()
df = pd.concat((df_new,df_0))
df.LABEL_COLUMN.value_counts()
X = df.loc[:,'DATA_COLUMN']
y = df.LABEL_COLUMN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=123)
X_train_docs = list(X_train)
X_test_docs =  list(X_test)

best_svc_model.fit(vect.fit_transform(X_train_docs), y_train)
X_test_features = vect.transform(X_test_docs)
y_test_pred = best_svc_model.predict(X_test_features)
accuracy_M4_U= metrics.accuracy_score(y_test, y_test_pred)
f1_score_M4_U= f1_score(y_test, y_test_pred)
print(f"The accuracy for Model 4 with undersampling is: {accuracy_M4_U:.2%} and f1 score is {f1_score_M4_U:.2%}")

# Oversampling
df,train,test,df_test,X_train_docs, y_train, X_test_docs, y_test,df_test_docs = refresh_data()

df_0 = df.loc[df['LABEL_COLUMN'] == 0,:]
df_1 = df.loc[df['LABEL_COLUMN'] == 1,:]
print(df_0.LABEL_COLUMN.value_counts())
print(df_1.LABEL_COLUMN.value_counts())
df_new= resample(df_0, replace=True, n_samples=df_1.shape[0], random_state=123)
df_new.LABEL_COLUMN.value_counts()
df = pd.concat((df_new,df_1))
df.LABEL_COLUMN.value_counts()
X = df.loc[:,'DATA_COLUMN']
y = df.LABEL_COLUMN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=123)
X_train_docs = list(X_train)
X_test_docs =  list(X_test)

best_svc_model.fit(vect.fit_transform(X_train_docs), y_train)
X_test_features = vect.transform(X_test_docs)
y_test_pred = best_svc_model.predict(X_test_features)
accuracy_M4_O = metrics.accuracy_score(y_test, y_test_pred)
f1_score_M4_O = f1_score(y_test, y_test_pred)
print(f"The accuracy for Model 4 with oversampling is: {accuracy_M4_O:.2%} and f1 score is {f1_score_M4_O:.2%}")

# BERT Model 
# The Training set accuracy for Model 5 is: 88.06% and f1 score is 92.65%
# The Testing set accuracy for Model 5T is: 85.52% and f1 score is 90.94%

# Refreshing and reassigning variables
df,train,test,df_test,X_train_docs, y_train, X_test_docs, y_test,df_test_docs = refresh_data()

# Using Bert's lowercased pretrained model
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# InputExamples are required for BERT - They take all the comments combined(data column) as text_a and the rating (1 or 0) as label
def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN): 
  train_InputExamples = train.apply(lambda x : InputExample(
    guid=None, # Globally unique ID for bookkeeping, unused in this case
    text_a = x[DATA_COLUMN], 
    text_b = None,
    label = x[LABEL_COLUMN]),axis = 1)

  validation_InputExamples = test.apply(lambda x: InputExample(
      guid=None, # Globally unique ID for bookkeeping, unused in this case
        text_a = x[DATA_COLUMN], 
        text_b = None,
        label = x[LABEL_COLUMN]), axis = 1)
  return train_InputExamples, validation_InputExamples

# Running the function and assigning variables
train_InputExamples, validation_InputExamples = convert_data_to_examples(
    train, 
    test, 
    'DATA_COLUMN', 
    'LABEL_COLUMN')

def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later
    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a, #Training text
            add_special_tokens=True, #SEP/CLS etc
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, 
            truncation=True)
        # Token type id = represented as a binary mask identifying the two types of sequence in the model.
        # Input ID = ID of each token represented by a number
        # Attention mask = indicates to the model which tokens should be attended to, and which should not. (as padding is applied)
        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])
        # Adding feature for each sentence through its label and the vectors listed above
        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )
    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )
    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

DATA_COLUMN = 'DATA_COLUMN'
LABEL_COLUMN = 'LABEL_COLUMN'

train_InputExamples, validation_InputExamples = convert_data_to_examples(
     train, 
     test, 
     'DATA_COLUMN', 
     'LABEL_COLUMN')

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data, epochs=2, validation_data=validation_data)

# Save model
# model.save_pretrained('BertModel')
# tokenizer.save_pretrained('tokenizer_bertModel')

model = TFBertForSequenceClassification.from_pretrained('BertModel')
# Load the saved tokenizer
tokenizer = BertTokenizer.from_pretrained('tokenizer_bertModel')

# Checking accuracy
test_InputExamples = test.apply(lambda x: InputExample(guid=None, text_a=x['DATA_COLUMN'], text_b=None, label=x['LABEL_COLUMN']), axis=1)
test_data = convert_examples_to_tf_dataset(list(test_InputExamples), tokenizer)
test_data = test_data.batch(32)
# Get predictions
y_pred = model.predict(test_data)
y_pred = tf.argmax(y_pred[0], axis=1).numpy()
# Get actual labels
y_test = test['LABEL_COLUMN'].tolist()
# Calculate F1 score
f1_score_M5 = f1_score(y_test, y_pred)
accuracy_M5 = metrics.accuracy_score(y_test, y_pred)

print(f"The accuracy for Model 5 is: {accuracy_M5:.2%} and f1 score is {f1_score_M5:.2%}")

# Testing Set Accuracy
# Convert the test data to InputExamples and tokenize it
test_InputExamples_T = df_test.apply(lambda x: InputExample(guid=None, text_a=x['DATA_COLUMN'], text_b=None, label=x['LABEL_COLUMN']), axis=1)
test_data_T = convert_examples_to_tf_dataset(list(test_InputExamples_T), tokenizer)
test_data_T = test_data_T.batch(32)

# Get predictions and calculate scores
y_pred_T = model.predict(test_data_T)
y_pred_T = tf.argmax(y_pred_T[0], axis=1).numpy()
y_test_T = df_test['LABEL_COLUMN'].tolist()
f1_score_M5_T = f1_score(y_test_T, y_pred_T)
accuracy_M5_T = metrics.accuracy_score(y_test_T, y_pred_T)
print(f"The testing set accuracy for Model 5 is: {accuracy_M5_T:.2%} and f1 score is {f1_score_M5_T:.2%}")

# For Streamlit Barchart
pd.Series(y_test_T).value_counts(normalize=True)


