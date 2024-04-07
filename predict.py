import joblib
import numpy as np
import pandas as pd

class Predict:
    MODEL_OUTPUT_PATH = "./output/"

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

    FEATURES_NAMES = ['step','isFlaggedFraud','oldbalanceOrg','newbalanceOrig','amount','oldbalanceDest','newbalanceDest','type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER','nameOrig_C','nameDest_C','nameDest_M']


    def predict(self):
        clf = joblib.load(self.MODEL_OUTPUT_PATH + "rf_model.sav")
        preprocessed_transaction_df = pd.DataFrame(self.PREDICT_INPUT, columns=self.FEATURES_NAMES)

        return clf.predict(preprocessed_transaction_df)

predict = Predict()
print(predict.predict())