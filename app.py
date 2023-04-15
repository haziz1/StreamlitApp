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
warnings.filterwarnings("ignore", message="Some layers from the model checkpoint at BertModel were not used when initializing TFBertForSequenceClassification.*")
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
import streamlit as st
import plotly.graph_objects as go
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.image as mpimg
# from drug_named_entity_recognition import find_drugs
# import joblib

# Defining Dataset
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Defining Labels
df['LABEL_COLUMN'] = df.rating.apply(lambda x : 1 if x >=5 else 0)
df_test['LABEL_COLUMN'] = df_test.rating.apply(lambda x : 1 if x >=5 else 0)
df['DATA_COLUMN'] = df['benefits_review'] + " " + df['side_effects_review'] + " " +df ['comments_review']
df_test['DATA_COLUMN'] = df_test['benefits_review'] + "" + df_test['side_effects_review'] + "" +df_test['comments_review']
benefitsvect = list(df_test.benefits_review)
sideeffectsvect = list(df_test.side_effects_review)
commentsvect = list(df_test.comments_review)
X = df.loc[:,'DATA_COLUMN']
y = df.LABEL_COLUMN
# Test & Training datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=123)
# Converting DF into list for analysis
X_train_docs = list(X_train)


# Defining header and paragraph fonts and size

def Header(text):
    st.markdown(f'<p style="color:#1F1F5E;font-size:45px;font-weight:bold;font-family:Helvetica;border-radius:2%;">{text}</p>', unsafe_allow_html=True)

def indentp(text, padding=0):
    style = f'font-size:16px; padding-left:{padding}em;'
    st.write(f'<span style="{style}">{text}</span>', unsafe_allow_html=True)
def paragraph(text):
    st.write(f'<span style="font-size:16px;">{text}</span>', unsafe_allow_html=True)


# Streamlit App 

st.set_page_config(layout="wide")

# Setting up tabs
A,B,C,D = st.tabs(["Introduction", "Modelling & Results", "Summary & Testing","Next Steps"])

with A:
    Header("Sentiment Analysis for Prescription Drugs Reviews")
    # st.text(" ")
    st.subheader("Creating a Machine Learning model that uses consumer comments to perform sentiment analysis")
    paragraph("This sentiment model uses a dataset that provides patient reviews on various prescription drugs related to different health conditions. Patient responses are recorded on three key aspects - benefits, side effects and overall comments. Additionally, ratings are available concerning overall satisfaction of the prescribed drug. (Credit: Drug Review Dataset on UCI Machine Learning Repository)")
    st.subheader("Sentiment Overview")
    paragraph("The model predicts most of the drug reviews to be positive which hints that customers are generally happy with their medication")
    # testing dataset chart
    data_dict = {
    'Sentiment': ['Positive', 'Negative'],
    'Proportion': [76.7, 23.2]}
    chart_data = pd.DataFrame.from_dict(data_dict)
    fig = px.bar(chart_data, x='Sentiment',y= 'Proportion',color = 'Sentiment',color_discrete_map={"Positive": "darkgreen", "Negative": "tomato"})
    fig.update_layout(width=600, height=400)
    # plot the chart using Streamlit
    st.plotly_chart(fig)

    # Following code saves the wordclouds for later
    # # Use Spacy's Medium dictionary
    # en_nlp = spacy.load("en_core_web_md")

    # # # Remove the digits from the features
    # pattern = re.compile(r'(?u)\b(?![\d_])\w+\b') 

    # def custom_tokenizer(document):
    #      doc_spacy = en_nlp(document)
    #      lemmas = [token.lemma_ for token in doc_spacy]
    #      return [token for token in lemmas if token not in STOP_WORDS and pattern.match(token) 
    #              and len(token) >= 2]

    # vect = TfidfVectorizer(tokenizer=custom_tokenizer, ngram_range=(1, 3), max_features=1000)

    # feature_names_benefits = vect.fit(benefitsvect).get_feature_names_out()
    # feature_names_sideeffec = vect.fit(sideeffectsvect).get_feature_names_out()
    # feature_names_comment = vect.fit(commentsvect).get_feature_names_out()

    # def generate_wordcloud(features,filename):
    #     wordcloud = WordCloud(width = 800, height = 800, background_color ='white', 
    #                         min_font_size = 10).generate(' '.join(features))
    #     plt.figure(figsize = (8, 8), facecolor = None) 
    #     plt.imshow(wordcloud) 
    #     plt.axis("off") 
    #     plt.tight_layout(pad = 0) 
    #     plt.savefig(filename, dpi=300)
    #     plt.show()
    # generate_wordcloud(feature_names_benefits,'benefits')
    # generate_wordcloud(feature_names_sideeffec,'sideeffects')
    # generate_wordcloud(feature_names_comment,'comments')
    # wordcloud_img3 = mpimg.imread('comments.png')
    # img1, img2 = st.columns([2,2])
    # with img1:
    #     wordcloud_img = mpimg.imread('benefits.png')
    #     plt.imshow(wordcloud_img)
    #     plt.axis('off')
    # with img2:
    #     wordcloud_img2 = mpimg.imread('sideeffects.png')
    #     plt.imshow(wordcloud_img2)
    #     plt.axis('off')
    st.subheader("Features of the Testing Dataset")
    paragraph("Extracting important features from the benefits and side effects columns reveal the following")
    st.text("")
    wordcloud_img = mpimg.imread('benefits.png')
    wordcloud_img2 = mpimg.imread('sideeffects.png')
    col1, mid, col2 = st.columns([5,0.4,5])
    with col1:
        st.image(wordcloud_img, width=500,use_column_width=True,caption = 'Benefits Review')
    with col2:
        st.image(wordcloud_img2,width =500,use_column_width=True,caption = 'Side Effects Review')

    col1, mid, col2 = st.columns([5,0.4,5])
    with col1:
        paragraph("Some of the benefits that customers reported from medication were :  ")
        indentp("1 - Reducing acne, pain & inflamation", padding = 2)
        indentp("2 - Improving mood, allergies & sleep",padding = 2)
        indentp("3- Decrease in pain, hairloss & soreness",padding = 2)
    with col2:
        paragraph("Some of the side effects that customers reported from medication were :  ")
        indentp("1 - Increase in weight, flakyness & dryness", padding = 2)
        indentp("2 - Nausea, dizziness & tenderlessness",padding = 2)
        indentp("3 - Tiredness, irritation & anxiety ",padding = 2)

    texts = list(df_test.DATA_COLUMN)

    st.write("")
    st.write("")
    paragraph("As most customers complain about anxiety its not surprising that the frequently mentioned drugs are: ")
    indentp(" 1- <b>Venlafaxine</b> : Antidepressant and Nerve pain medication",padding = 2)
    indentp(" 2- <b>Escitalopram</b> : Depression and Anxiety medication",padding = 2)
    indentp(" 3- <b>Duloxetine</b> : Antidepressant and Nerve pain medication",padding = 2)
    st.write("")
    # drugs=[]
    # for text in texts:
    #     drugs_in_text = find_drugs(text.split(" "), is_ignore_case=True)
    #     for names in drugs_in_text:
    #         drugs.append(names[0]['name'])
    # generate_wordcloud(drugs,'drugnames')
    col1, mid, col2 = st.columns([3,7,1])
    with mid :
        wordcloud_img = mpimg.imread('drugnames.png')
        st.image(wordcloud_img, caption='Frequent Drug Names', use_column_width=False, width=550)



with B:
    st.subheader(":violet[Introducing the Dataset & Preprocessing]")
    indentp(f' Training dataset has <b> {df.shape[0]}</b> rows and <b>4 </b> columns')
    indentp('Following are some samples. *Do note that some of these columns are empty or lack important information. For example some rows have "None" or " - "*')

    df_last_10 = df.iloc[-10:, :]
    if st.button("Show sample data"):
        placeholder = st.empty()
        with placeholder.container():
            a, b, c, d = st.columns([2, 2, 2, 1])
            with a:
                st.write("**Benefits Review**")
            with b:
                st.write("**Side Effects Review**")
            with c:
                st.write("**Comments Review**")
            with d:
                st.write("**Rating**")
            for index, row in df_last_10.iterrows():
                a, b, c, d = st.columns([2, 2, 2, 1])
                text1 = row["benefits_review"]
                with a:
                    with st.expander(text1[:150] + "..."):
                        st.write(text1)
                text2 = row["side_effects_review"]
                with b:
                    with st.expander(text2[:150] + "..."):
                        st.write(text2)
                text3 = row["comments_review"]
                with c:
                    with st.expander(text3[:150] + "..."):
                        st.write(text3)
                with d:
                    st.write(row['rating'])
        if st.button("Clear", type="primary"):
                placeholder.empty()
    paragraph('Based on the above, it would be a good idea to :')
    indentp('1 - Combine all the text together into 1 final column',padding = 4)
    indentp('2 - Change all <b> ratings >= 5 </b> into <b>Positive</b> while those being below to <b>Negative</b> for sentiment analysis',padding=4)
    paragraph('The dataset would look something like this then: ')
    if st.button("Show Updated Dataset"):
        placeholder = st.empty()
        with placeholder.container():
            a, b= st.columns([4, 1])
            with a:
                st.write("**All Comments**")
            with b:
                st.write("**Sentiment**")
            for index, row in df.sample(10).iterrows():
                a, b= st.columns([4, 1])
                text1 = row["DATA_COLUMN"]
                with a:
                    with st.expander(text1[:150] + "..."):
                        st.write(text1)
                text2 = row["LABEL_COLUMN"]
                with b:
                    st.write(row['LABEL_COLUMN'])
        if st.button("Clear", type="primary"):
                placeholder.empty()

    st.write("")
    paragraph('The dataset looks imbalanced with most customers having Positive (1) reviews')
    # Imbalance in dataset
    proportion_series = (df.LABEL_COLUMN.value_counts(normalize=True)*100).apply(lambda x: '{:.2f}%'.format(x))
    proportion_df = pd.DataFrame(proportion_series).rename(columns={"LABEL_COLUMN": "Proportion"})
    st.table(proportion_df)

    st.subheader(":violet[Initializing Models]")
    paragraph('Using 2 different vectorizing techniques with a LinearSVC Model:')
    indentp('*Note - Testing accuracy refers to models performance on unseen training data. Secondly, each model is compared to its predecessor*')

    # 1- Training Dataset Accuracy = 75.91% & F1-Score = 84.27%
    # 2- Testing Dataset Accuracy = 75.68% & F1-Score = 83.99%  
    st.subheader("Model 1 - CountVectorizer")
    col1, col2, col3,col4,col5,col6 = st.columns(6)
    col1.metric("Cross Validation Score", "76.7%")
    col2.metric("Training Set Accuracy", "75.9%")
    col3.metric("Training Set F1 Score", "84.3%")
    col4.metric("Testing Set F1 Score", "75.7%")
    col5.metric("Testing Set F1 Score", "84.0%")
    with col6:
        'Specifications are as follows:  \n1 - N-grams : 1-3  \n2 - Max Features : 1000  \n3 - Removing stop words'

    # Model 2 - Improving model - Using Lemmitization 
    # 1- Training Dataset Accuracy = 79.35% & F1-Score = 87.30%
    # 2- Test Dataset Accuracy = 80.89% & F1-Score = 88.06%

    st.subheader("Model 2 - TfidfVectorizer")
    col1, col2, col3,col4,col5,col6 = st.columns(6)
    col1.metric("Cross Validation Score", "80.4%","3.7pp")
    col2.metric("Training Set Accuracy", "79.4%","3.5pp")
    col3.metric("Training Set F1 Score", "87.3%","3pp")
    col4.metric("Testing Set Accuracy", "80.9%","5.2pp")
    col5.metric("Testing Set F1 Score", "88%","4pp")
    with col6:
        "Specifications are as follows:  \n1 - N-grams : 1-2  \n2 - Max Features : 1000  \n3 - Removing stop words  \n4 - Lemmatization  \n5 - Spacy's small dictionary"

    paragraph('It is clear that TfidfVectorizer & Lemmitization perform better so lets optimize the model further')

    st.subheader(":violet[Improving TfidfVectorizer]")
    paragraph('To improve the model, we can optimize each step of Model 2')
    paragraph('This entails the following : ')
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        paragraph('<b>1 - Improving Tokenization - Removing digits from features and any single words that the model picks up such as "I" or "U" </b>')
        indentp(f"*For the curious the code look like this re.compile('(?u)\\b(?![\\d_])\\w+\\b')*")
    with col2:
        paragraph('<b>2 - Use GridSearchCV to find the best Machine Learning Model from the following : </b>')
        indentp('1 - Logistic Regression',padding=2)
        indentp('2 - Naïve Bayes',padding =2 )
        indentp('3 - LinearSVC',padding=2)
        indentp('4 - XGBClassifier',padding=2)
    with col3:
        paragraph('<b>3 - Hyperparameter tuning the winner - LinearSVC:</b>')
        indentp('1 - C: 1',padding=2)
        indentp('2 - dual: True',padding =2 )
        indentp('3 - loss: hinge',padding=2)
        indentp('4 - penalty: l2',padding=2)
        indentp('5 - tol: 0.0001',padding=2)
    with col4:
        paragraph('<b>4 - Optimize the tfidfVectorizer</b>')
        indentp('The optimal features for the vectorizer are:')
        indentp('1 - Max Features : 1000',padding =2 )
        indentp('2 - N-grams : 1,3',padding=2)

    paragraph('Putting it all together')

    st.subheader("Model 3 - TfidfVectorizer (but on steroids)")
    col1, col2, col3,col4,col5 = st.columns(5)
    col1.metric("Cross Validation Score","81.6%","1.2pp")
    col2.metric("Training Set Accuracy", "82.0%","2.6pp")
    col3.metric("Training Set F1 Score", "89.4%","2.1pp")
    col4.metric("Testing Set Accuracy", "81.2%","0.3pp")
    col5.metric("Testing Set F1 Score", "88.7%","0.7pp")

    st.text("")
    st.subheader(":violet[Hol' Up, didn't you tell me the dataset was imbalanced?]")
    paragraph("Balancing dataset is a double edged sword. While undersampling & oversampling can improve accuracy, it can also cause loss of information or overfitting.")

    st.subheader("Model 4A - Undersampling")
    col1,col2 = st.columns(2)
    with col1:
        col1.metric("Training Set Accuracy", "70.1%","-11.9pp")
    with col2:
        col2.metric("Training Set F1 Score", "70.5%","-18.9pp")

    st.subheader("Model 4B - Oversampling")
    col1,col2 = st.columns(2)
    with col1:
        col1.metric("Training Set Accuracy", "81%","-1pp")
    with col2:
        col2.metric("Training Set Score", "80.4%","-9pp")

    paragraph("As predicted, both techniques underperformed as compared to Model 3")

    st.subheader(":violet[Introducing the BERT Model]")
    paragraph("So far we have been tokenizing and using ngrams to predict sentiment. However, a critical part of natural language is understanding context & word associations before we predict sentiment.")
    paragraph("BERT - Bidirectional Encoder Representations from Transformers is a pre-trained transformer that has been trained on large corpus of text. It specializes in considering the context before and after a token to make predictions.")
    paragraph("It is hence not surprising that even a primitive BERT model outperforms the other models.")
    st.subheader("Model 5 - Bert Model")
    col1, col2, col3,col4,col5 = st.columns(5)
    col1.metric("Cross Validation Score","89.1%","7.5pp")
    col2.metric("Training Set Accuracy", "88.1%","6.1pp")
    col3.metric("Training Set F1 Score", "92.7","3.3pp")
    col4.metric("Testing Set Accuracy", "85.5%","4.3pp")
    col5.metric("Testing F1 Score", "90.9%","2.2pp")

with C:
    st.subheader(":violet[Summarizing Learnings]")
    indentp("1 - Bert performed clearly the best across both training and testing datasets (In both F1 and accuracy metrics).")
    indentp("2 - Model 3 performed well on training dataset but didn't perform better than Model 2 on testing dataset. This indicates that hyperparamter tuning doesn't result in major accuracy increases (albeit a small increase in f1).")
    st.subheader(":violet[Comparison of Model Metrics]")
    model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4A', 'Model 4B', 'Model 5']
    Testing_Set_Accuracies = [0.76, 0.81, 0.81, 0.70, 0.81, 0.86]
    Testing_Set_F1_Score = [.84, .88, .89, .71, .80, .91]
    df = pd.DataFrame({'Model': model_names, 'Accuracy': Testing_Set_Accuracies, 'F1 Score': Testing_Set_F1_Score})
    metric = st.selectbox('Select a metric', ['Accuracy', 'F1 Score'])
    if metric == 'Accuracy':
        df_metric = df[['Model', 'Accuracy']]
    else:
        df_metric = df[['Model', 'F1 Score']]

    # Create the bar chart using Altair
    chart = alt.Chart(df_metric).mark_bar(color='purple').encode(
        x=alt.X('Model', axis=alt.Axis(labelAngle=0)),
        y=alt.Y(metric, title=metric),
        tooltip=['Model', metric]
    ).properties(
        width=600,
        height=400,
        title='Comparison of Model {}s'.format(metric)
    )
    # Display the chart on Streamlit
    st.altair_chart(chart)

    model = TFBertForSequenceClassification.from_pretrained('BertModel')
    # Load the saved tokenizer
    tokenizer = BertTokenizer.from_pretrained('tokenizer_bertModel')

    st.subheader(":violet[Testing the Model]") 
    paragraph("Want to test out the model yourself?")
    paragraph(" Type out any sentence from the model, our suggested sentences or if you're feeling lucky any sentence of your own! The model will try to predict the sentiment of the sentence")

    user_input = st.text_input("Enter your sentence: ")
    if user_input == "":
        st.write('No input. Please enter a sentence to test out the model')
    else : 
        # Tokenize the input
        tf_batch = tokenizer(user_input, max_length=128, padding=True, truncation=True, return_tensors='tf')
        # Get the predicted label
        tf_outputs = model(tf_batch)
        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
        labels = ['Negative','Positive']
        label_idx = tf.argmax(tf_predictions, axis=1)
        label = labels[label_idx.numpy()[0]]
        # Print the predicted sentiment
        st.write(f"The sentiment is: {label}")
    st.write("")
    st.write("")
    paragraph('Cant think of any examples right now? Try some of our recommended trickier AI generated reviews:')
    indentp('1 - The medication helped with my symptoms, but the side effects were unbearable',padding = 2)
    indentp('2 - I have been taking this medication for a week and have noticed no improvement',padding = 2)
    indentp('3 - I was initially hesitant to try this medication, but it has helped me manage my chronic pain better than anything else I have tried',padding = 2)

with D:
    st.subheader(":violet[Mapping further improvements]")
    indentp("As with any model, there are always a number of improvements that can be made to enchance accuracy:")
    indentp("1 - Increase text corpus for analysis",padding =2)
    indentp("2 - Testing the model by setting the threshold for Positive at a higher rating such as 7/8",padding = 2)
    indentp("3 - Splitting sentiment into Negative, Neutral & Positive to enchance quality of predicted sentiment",padding = 2)
    indentp("4 - Hyperparameter tuning the BERT model with greater number of features and sentence length input (Currently trained on max length of 128)",padding = 2 )
    indentp("5 - Mapping drug names to sentiment to analyze drugwise sentiment",padding = 2) 
    st.text("")
    indentp("Incorporating the above should improve the accuracy, reliability and usefulness to the end users ")
# streamlit run app.py --server.runOnSave true