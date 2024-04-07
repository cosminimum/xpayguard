# Card Fraud Detection System: Proof of Concept

This repository houses the proof of concept for a fraud detection system, 
as proposed in [recent thesis](https://docs.google.com/document/d/1H6mYeO212AjY1j5StG5QG-mZkEY11btcCxAnB8i-l6Y/edit?usp=sharing). 

Designed to address the complex and evolving nature of 
fraudulent activities, this project utilizes advanced 
machine learning techniques to identify and prevent 
fraud in real-time. 

This proof of concept demonstrates the 
practical applicability and effectiveness of my 
proposed methods in detecting fraud across ISO8583 transactions.

## Premises
1. Consider as input a dataset with unique values and not null data
2. Most of the data is not correlated
3. The data does not have normal distribution
4. Timeframe is about 744 steps (30 days)

## Data Privacy

Dataset has the following features:

- step - unit of time - 1 step = 1 hour
- type - transaction type
- amount - transaction amount
- nameOrig - actor who started transaction
- oldbalanceOrg - initial balance before the transaction
- newbalanceOrig - new balance after the transaction
- nameDest - recipient actor of the transaction
- oldbalanceDest - initial balance recipient before the transaction
- newbalanceDest - new balance recipient after the transaction
- isFraud - fraud detection flag
- isFlaggedFraud - flag for transactions over 200k

### What about privacy?
For our testing purposes, we will use non anonymously data, because
it's important to validate our premisese.

In real life scenario, DataLayer will manipulate data and will
simplify through [PCA Transformation](https://en.wikipedia.org/wiki/Principal_component_analysis)

In this way our data will look like:
```
"Step","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"
0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62,"0"
```

Where `V++` are our anonymised features, and `step` and `amount` will remain visible. 

## Imbalanced Classes

In the dataset utilized for our fraud detection system, we encounter a significant challenge common to many real-world applications: imbalanced classes. Specifically, our dataset comprises:

- 1,270,904 non-fraudulent transactions
- 1,620 fraudulent transactions

Majority of non-fraud transactions can lead to a model that performs poorly by simply 
predicting the majority class for all inputs.

To address this issue, I implement [SMOTE](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)

SMOTE helps to create a more balanced dataset, which in turn facilitates the development of a model that can identify fraud more accurately without being biased towards the majority class.

### Exploratory: How SMOTE Works with Generative AI [TBD]

### Accuracy

Check accuracy:
```
score = clf.score(x_test,y_test)
y_preds = clf.predict(x_test)
score, y_preds
```


For our dataset, accuracy is `99.6%`

### Feature Importance
Check feature importance:
```
importances = clf.feature_importances_
forest_importances = pd.Series(importances, index=feature_names)
```

For our dataset, feature importance is as follows:
```
# oldbalanceOrg     0.270654
# amount            0.154807
# newbalanceOrig    0.120159
# step              0.089747
# newbalanceDest    0.082174
# type_TRANSFER     0.080242
# oldbalanceDest    0.056131
# type_CASH_OUT     0.034524
# type_CASH_IN      0.033863
# nameDest_C        0.027732
# nameDest_M        0.025882
# type_PAYMENT      0.023631
# type_DEBIT        0.000447
# isFlaggedFraud    0.000008
# nameOrig_C        0.000000
```

# How to start

1. [Download dataset](https://drive.google.com/file/d/1qU3piT1pAfYRoE1OG8LKFplJ1dZB0BNR/view?usp=sharing)
2. Copy to `/input`
3. Run `python3 ./model.py`
4. For prediction, run `python3 ./train.py`

# How to predict

## Input Format
```
PREDICT_INPUT = {
        'step': [1],
        'isFlaggedFraud': [0],
        'oldbalanceOrg': [np.log(170136.0 + 1)],
        'newbalanceOrig': [np.log(160296.36 + 1)],
        'amount': [np.log(9839.64 + 1)],
        'oldbalanceDest': [np.log(0.0 + 1)],
        'newbalanceDest': [np.log(0.0 + 1)],
        'type_CASH_IN': [0],
        'type_CASH_OUT': [0],
        'type_DEBIT': [0],
        'type_PAYMENT': [1],
        'type_TRANSFER': [0],
        'nameOrig_C': [1],
        'nameDest_C': [0],
        'nameDest_M': [1]
   }
```

## Expected Output
```
[0] - non-fraudulent transaction
[1] - fraudulent transaction
```

# References
1. [Random Forest Classifier + Feature Importance](https://www.kaggle.com/code/prashant111/random-forest-classifier-feature-importance)
2. [Credit Fraud || Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/notebook)
3. [Credit Card Fraud Detection Predictive Models](https://www.kaggle.com/code/gpreda/credit-card-fraud-detection-predictive-models/input)




