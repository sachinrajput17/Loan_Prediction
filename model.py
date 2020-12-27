import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier



# Train Data
loan_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )
loan_data.iloc[:,2:].to_csv("loan_data.csv",index = False)
data = loan_data.iloc[:,2:]

numeric = ['ApplicantIncome','CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']
cat = ['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Property_Area','Credit_History']


sc = StandardScaler()
imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')
imputer_cat= SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
imputer.fit(data[numeric])
imputer_cat.fit(data[cat])
data[numeric] = imputer.transform(data[numeric])
data[cat] = imputer_cat.transform(data[cat])


le = LabelEncoder()
for col in cat:
  data[col]= le.fit_transform(data[col])

X = data.drop("Loan_Status",axis=1)
y = data["Loan_Status"]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 101,test_size=0.3)

log= LogisticRegression(random_state = 1,penalty = 'l1',solver = 'saga')
dt = DecisionTreeClassifier(random_state = 1)
rf = RandomForestClassifier(max_depth = 7,random_state=1)
xgb = XGBClassifier(max_depth=3,min_child_weight=5)

def score(model):
  print("___________________________________________________________________________________\n")
  print("Training F1 Score",f1_score(y_train,model.predict(X_train)))
  print("___________________________________________________________________________________\n")
  print("Testing F1 Score",f1_score(y_test,model.predict(X_test)))
  print("___________________________________________________________________________________\n")
  

def fit(model):
  model.fit(X_train,y_train)
  score(model)
  
fit(rf)
fit(xgb)


def Predict(model,data):
  return model.predict(data)  
  
import pickle
filename = 'model.pkl'
pickle.dump(xgb, open(filename, 'wb'))
  
  
  
  


