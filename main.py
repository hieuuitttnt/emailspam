from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import category_encoders as ce

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
data = pd.read_csv(r"C:\E disk\tailieu sinh vien\ML\archive\emails.csv")
#data[0:3]

class Email(BaseModel):
    message: str

app = FastAPI()
target = 'Prediction'
labels = ['Ham','Spam']
features = [i for i in data.columns.values if i not in [target]]
original_df = data.copy(deep=True)
original_df = data.copy(deep=True)
counter = 0
r,c = original_df.shape

df1 = data.copy()
df1.drop_duplicates(inplace=True)
df1.reset_index(drop=True,inplace=True)
nvc = pd.DataFrame(df1.isnull().sum().sort_values(), columns=['Total Null Values'])
nvc['Percentage'] = round(nvc['Total Null Values']/df1.shape[0],3)*100
data=data.set_index('Email No.', drop=True)
n=len(data)
train = data[0:(n//10)*8]
test = data[(n//10)*8:]
y_train0 = train['Prediction']
y_test0 = test['Prediction']
X_train0 = train.drop('Prediction', axis = 1)
X_test0 = test.drop('Prediction', axis = 1)
X = np.array(X_train0)
y = np.array(y_train0)
clf = XGBClassifier(max_depth=3, n_estimators=1000, learning_rate=0.01)
ss = ShuffleSplit(n_splits=5,train_size=0.8,test_size =0.2,random_state=0) 

for train_index, test_index in ss.split(X): 

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]

    clf.fit(X_train, Y_train) 
    

@app.post("/predict")
def predict_spam(email: Email):
    # Preprocess the input message
    preprocessed_message = email.message # Replace 'preprocess_message' with your actual preprocessing logic

    # Make the prediction using the loaded model
    prediction = clf.predict(preprocessed_message)

    # Return the predicted label
    return {"prediction": prediction}