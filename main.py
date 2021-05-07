import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



dataset_name = st.sidebar.selectbox("Select Dataset", ("Breast Cancer Prediction", "Heart Disease Prediction",
                                                       "Social_Network_Ads", "Person Body Type"))

# Importing the datasets :
def get_dataset(dataset_name):
    if dataset_name == "Breast Cancer Prediction":
        dataset = pd.read_csv('Data/Breast Cancer.csv')
        X = dataset.iloc[:, 1:-1].values
        y = dataset.iloc[:, -1].values
    elif dataset_name == "Heart Disease Prediction":
       dataset = pd.read_csv('Data/heart.csv')
       X = dataset.iloc[:, :-1].values
       y = dataset.iloc[:, -1].values
    elif dataset_name == "Social_Network_Ads":
       dataset = pd.read_csv('Data/Social_Network_Ads.csv')
       X = dataset.iloc[:, :-1].values
       y = dataset.iloc[:, -1].values
       
    elif dataset_name == "Person Body Type":
       dataset = pd.read_csv('Data/500_Person_Gender_Height_Weight_Index.csv')
       X = dataset.iloc[:, :-1].values
       y = dataset.iloc[:, -1].values
       from sklearn.preprocessing import LabelEncoder
       le = LabelEncoder()
       X[:, 0] = le.fit_transform(X[:, 0])
       
    return X, y


X, y = get_dataset(dataset_name)
st.title(f'{dataset_name}')
st.sidebar.write("shape of dataset", X.shape)
st.sidebar.write("number of classes", len(np.unique(y)))
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "kernel SVM", "Naive Bayes",
                                                             "Decision Tree", "Random Forest", "XGBoost"))

def dataset_input(dataset_name):
    inputs = dict()
    if dataset_name == "Person Body Type":
        user_input1=st.slider("sex (1 = male; 0 = female)", 0, 1, step=1)
        user_input2=st.slider("Height", 130, 220, step=1)
        user_input3=st.slider("Weight", 40, 170, step=1) 
        inputs=[[user_input1, user_input2, user_input3]]
                
    if dataset_name == "Social_Network_Ads":
        st.subheader("Predict weather the customer bought the item or not (could use this info to choose the best customer to advertise)")
        user_input1=st.slider("Age", 18, 70, step=1)
        user_input2=st.slider("Estimated Salary", 15000, 150000, step=100)
        inputs=[[user_input1, user_input2]]
    
    if dataset_name == "Breast Cancer Prediction":
        st.subheader("Predict weather the tumor is Benign(normal) or Malignant(abnormal)")
        user_input1=st.slider("Clump Thickness", 1, 10, step=1)
        user_input2=st.slider("Uniformity of Cell Size", 1, 10, step=1)
        user_input3=st.slider("Uniformity of Cell Shape", 1, 10, step=1)
        user_input4=st.slider("Marginal Adhesion", 1, 10, step=1)
        user_input5=st.slider("Single Epithelial Cell Size", 1, 10, step=1)
        user_input6=st.slider("Bare Nuclei", 1, 10, step = 1)
        user_input7=st.slider("Bland Chromatin", 1, 10, step=1)
        user_input8=st.slider("Normal Nucleoli", 1, 10, step=1)
        user_input9=st.slider("Mitoses", 1, 10, step = 1)
        inputs=[[user_input1, user_input2, user_input3, user_input4, user_input5,
         user_input6, user_input7, user_input8, user_input9]]
        
    if dataset_name == "Heart Disease Prediction":
        st.subheader("Predict if the patient has a heart disease")
        user_input1=st.slider("age", 20, 80, step=1)
        user_input2=st.slider("sex (1 = male; 0 = female)", 0, 1, step=1)
        user_input3=st.slider("chest pain type", 0, 3, step=1)
        user_input4=st.slider("resting blood pressure in mm/Hg", 90, 210, step=1)
        user_input5=st.slider("serum cholestoral in mg/dl", 120, 600, step=1)
        user_input6=st.slider("fasting blood sugar > 120 mg/dl", 0, 1, step = 1)
        user_input7=st.slider("resting electrocardiographic results", 0, 2, step=1)
        user_input8=st.slider("maximum heart rate achieved", 70, 210, step=1)
        user_input9=st.slider("exercise induced angina (1 = Yes; 0 = No)", 0, 1, step = 1)
        user_input10=st.slider("oldpeak = ST depression induced by exercise relative to rest", 0.0, 7.0, step=0.1)
        user_input11=st.slider("the slope of the peak exercise ST segment", 0, 2, step = 1)
        user_input12=st.slider("number of major vessels colored by flourosopy", 0, 4, step = 1)
        user_input13=st.slider("thal: ( 3 = normal; 6 = fixed defect; 7 = reversable defect )", 0, 3, step = 1)
        inputs=[[user_input1, user_input2, user_input3, user_input4, user_input5,
         user_input6, user_input7, user_input8, user_input9, user_input10, user_input11,
         user_input12, user_input13]]
    return inputs

inputs = dataset_input(dataset_name)

    
# Adding the models parameters :
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
        
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
        
    elif clf_name == "kernel SVM":
        kernel = st.sidebar.selectbox(("Choose Kernel function"), ('rbf', 'sigmoid', 'poly', 'linear'))
        C = st.sidebar.slider("C", 0.01, 10.0)
        params['kernel'] = kernel
        params["C"] = C
        
    elif clf_name =="Decision Tree":
        criterion = st.sidebar.selectbox(("Choose a Criterion method"), ('gini', 'entropy'))
        max_depth = st.sidebar.slider("max_depth", 1, 10)
        params['criterion'] = criterion
        params["max_depth"] = max_depth
        
    elif clf_name == "Random Forest":
        criterion = st.sidebar.selectbox(("Choose a Criterion method"), ('gini', 'entropy'))
        n_estimators = st.sidebar.slider("n_estimators", 100, 1000)
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params['criterion'] = criterion
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
        
    elif clf_name == "XGBoost":
        n_estimators = st.sidebar.slider("n_estimators", 100, 1000)
        max_depth = st.sidebar.slider("max_depth", 1, 10)
        learning_rate = st.sidebar.slider("learning_rate", 0.1, 0.01)
        subsample = st.sidebar.slider("subsample", 0.5, 1.0)
        min_child_weight = st.sidebar.slider("min_child_weight", 1, 10)
        colsample_bytree = st.sidebar.slider("colsample_bytree", 0.3, 1.0)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
        params["learning_rate"] = learning_rate
        params["subsample"] = subsample
        params["min_child_weight"] = min_child_weight
        params["colsample_bytree"] = colsample_bytree
        
    return params

params = add_parameter_ui(classifier_name)


# Choosing the classifiers :
def get_classifier(clf_name, params):
    if clf_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=params["K"])
        
    elif clf_name == "SVM":
               clf = SVC(C=params["C"])
        
    elif clf_name == "kernel SVM":
              clf = SVC(kernel = params['kernel'], 
                    C = params["C"], random_state = 0)
        
    elif clf_name == "Naive Bayes":
               clf = GaussianNB()
               
    elif clf_name == "Decision Tree":
        clf = DecisionTreeClassifier(criterion = params['criterion'],
                                     max_depth = params['max_depth'], random_state = 0)
                  
        
    elif clf_name == "Random Forest":
              clf = RandomForestClassifier(n_estimators = params["n_estimators"],
                                           criterion = params['criterion'],
                                           max_depth = params["max_depth"], random_state = 0)
              
    elif clf_name == "XGBoost":
              clf = XGBClassifier(max_depth = params["max_depth"],
                                  n_estimators = params["n_estimators"],
                                  learning_rate = params["learning_rate"],
                                  subsample = params["subsample"],
                                  min_child_weight = params["min_child_weight"],
                                  colsample_bytree = params["colsample_bytree"])
    return clf

clf = get_classifier(classifier_name, params)

#Splitting the dataset into the Training set and Test set :
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training the model on the training set :
clf.fit(X_train, y_train)

#Making the perdictions :
user_pred = clf.predict(sc.transform(inputs))
print(user_pred)

if dataset_name == "Breast Cancer Prediction":
    if user_pred == 2:
        st.info("Benign")
    elif user_pred == 4:
        st.warning("Malignant")
if dataset_name == "Heart Disease Prediction":
    if user_pred == 0:
        st.info("Absence of Heart Disease")
    elif user_pred ==1:
        st.warning("Presence of Heart Disease")
if dataset_name == "Social_Network_Ads":
    if user_pred == 1:
        st.info("Purchased the item before")
    elif user_pred == 0:
        st.warning("Didn't purchase the item before")
if dataset_name == "Person Body Type":
    if user_pred == 0:
        st.error("Extremely Weak")
    elif user_pred == 1:
        st.info("Weak")
    elif user_pred == 2:
        st.success("Normal")
    elif user_pred == 3:
        st.info("Overweight")
    elif user_pred == 4:
        st.warning("Obesity")
    elif user_pred == 5:
        st.error("Extreme Obesity")
        
        
#Showing the accuaracy results
from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.sidebar.info(f'Accuracy = {acc}')



    
         

