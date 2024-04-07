import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.special import boxcox1p
import re
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class Model:
    DTYPES = {
        'step':'int32',
        'type':'category',
        'amount':'float32',
        'oldbalanceOrg':'float32',
        'newbalanceOrig':'float32',
        'oldbalanceDest':'float32',
        'newbalanceDest':'float32',
        'isFraud':'int32',
        'isFlaggedFraud':'int32',
    }

    FEATURES = [
        'step',
        'isFlaggedFraud',
        'oldbalanceOrg',
        'newbalanceOrig',
        'amount',
        'oldbalanceDest',
        'newbalanceDest',
        'type_CASH_IN',
        'type_CASH_OUT',
        'type_DEBIT',
        'type_PAYMENT',
        'type_TRANSFER',
        'nameOrig_C',
        'nameDest_C',
        'nameDest_M'
    ]

    DATA_INPUT_PATH = "./input/sample.csv"
    MODEL_OUTPUT_PATH = "./output/"

    KNOWN_SKEWED_COLS = [
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest'
    ]


    def train(self):
        df = pd.read_csv(self.DATA_INPUT_PATH, dtype = self.DTYPES)

        # As I said in README, our values are not correlated
        # So it's important to discover what columns are not correlated and reduce the skew between them
        df_logs = df[self.KNOWN_SKEWED_COLS]
        df_boxcox = df[self.KNOWN_SKEWED_COLS]

        for i in self.KNOWN_SKEWED_COLS:
          df_logs.loc[:,i] = np.log(df_logs[i]+1)
          df_boxcox.loc[:,i] = boxcox1p(df_boxcox[i], 0.1)

        df = pd.concat([
            df[['step','type','nameOrig', 'nameDest','isFraud', 'isFlaggedFraud']],
            df[self.KNOWN_SKEWED_COLS][['oldbalanceOrg','newbalanceOrig']],
            df[self.KNOWN_SKEWED_COLS][['amount','oldbalanceDest','newbalanceDest']]
            ], axis=1, join='inner')

        # Transform all customers / merchants ID in M/C constants. We need to trace type, not entity itself
        df['nameOrig'] = df['nameOrig'].apply(lambda x: re.sub('C[A-Za-z0-9]*', 'C',x))
        df['nameDest'] = df['nameDest'].apply(lambda x: re.sub('C[A-Za-z0-9]*','C',x))
        df['nameDest'] = df['nameDest'].apply(lambda x: re.sub('M[A-Za-z0-9]*','M',x))

        ohe = preprocessing.OneHotEncoder()
        df= pd.get_dummies(df, columns=['type','nameOrig', 'nameDest'])

        target=['isFraud']
        x_train, x_test, y_train, y_test = train_test_split(df[self.FEATURES], df['isFraud'], test_size=0.2, random_state=42)

        over = SMOTE(sampling_strategy=0.025)
        under = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
        steps = [
            ('o', over),
            ('u', under)
        ]

        pipeline = Pipeline(steps=steps)

        x_res, y_res = pipeline.fit_resample(x_train[self.FEATURES], y_train)

        clf = RandomForestClassifier(criterion = 'gini', max_depth = 48, max_features = 'sqrt', n_estimators = 300)

        clf.fit(x_res, y_res)

        joblib.dump(clf, self.MODEL_OUTPUT_PATH + "rf_model.sav")


model = Model()
model.train()