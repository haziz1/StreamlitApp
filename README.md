# StreamlitApp

Using the drug review dataset, the model and app aim to predict consumer sentiment from new customer reviews. The model takes 3 columns as features namely customer benefit reviews, side effect reviews and final comment reviews to predict rating, a column that has been converted into a binary 1 (Positive) and 0 (Negative) for analysis. 

Based on the above, the model predicts that most consumer sentiments are positive and relate to problems with depression and medication. Hence, it is not surprising that the most frequently referred drugs are antidepressants.

The model and app explore 5 different models to increase accuracy of output. Models 1 and 2 use basic Count and Tfidf vectorizers while model 3 uses hyper parameter tuning to optimize accuracy and f1 score. As the initial dataset is imbalanced, model 4 explores different sampling techniques such as under and over sampling to correct for imbalance while model 5 uses a pre trained Bert model to analyze training data.

Results from the modelling indicate that under and oversampling reduce accuracy and hyperparameter tuning does not improve accuracy and f1 score by a significant amount. Bert model performs the best across all datasets and is the final model chosen for the analysis.

Furthermore, the app lets the user test their own reviews or comments to predict sentiment from their text. Finally, next steps for the model include increasing text corpus, using more features for the Bert model, including a neutral sentiment as a predictor and changing the ratings for each sentiment (0 – 4 for negative, 5 – 7 for neutral and 8 -10 for positive)

