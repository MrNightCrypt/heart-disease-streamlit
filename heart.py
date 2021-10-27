# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:12:42 2021

@author: Hein Htet Hlaing
"""

import streamlit as st
import base58
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.metrics import confusion_matrix
import pickle as pkl
from pyngrok import ngrok

st.set_page_config(
	layout="centered",
	initial_sidebar_state="expanded",
	page_title="Heart Disease Detector",	
    page_icon="⚕",
)
	

#Machine Learning

heart_disease = pd.read_csv(r"heart.csv")
#heart_disease.head()

#heart_disease.info()

heart_disease.isna().sum()

sex_count = heart_disease.sex.value_counts()
#sex_count

sex_count_sns = sns.countplot('sex', data = heart_disease)

hd_sex_count = heart_disease.sex[heart_disease.target==1].value_counts()
#hd_sex_count

chart1 = hd_sex_count.plot(kind='bar',figsize=(10,6),color=['green','blue'])
plt.title("Count of the number of males and females with heart disease")
plt.xticks(rotation=0);

fig1 = pd.crosstab(heart_disease.target, heart_disease.sex)

chart2 = fig1.plot(kind='bar',figsize=(10,6),color=["lightblue","pink"])
plt.title("Frequency of Heart Disease vs Sex")
plt.xlabel("0= Heart Disease, 1= No disease")
plt.ylabel("Number of people with heart disease")
plt.legend(["Female","Male"])
plt.xticks(rotation=0);

cor_mat=heart_disease.corr()
fig,ax=plt.subplots(figsize=(10,6))
sns.heatmap(cor_mat,annot=True,linewidths=0.5,fmt=".3f")

scal=MinMaxScaler()
feat=['age', 	'sex', 	'cp', 'trestbps', 'chol', 	'fbs', 	'restecg', 	'thalach' ,	'exang', 	'oldpeak' ,	'slope', 	'ca', 'thal']
heart_disease[feat] = scal.fit_transform(heart_disease[feat])
#heart_disease.head()

#number of heart disease and Gender wise
hd_sex = sns.countplot('sex',hue='target',data=heart_disease)


scaler = StandardScaler()
features= ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
#heart_disease.head()

X=heart_disease.drop("target",axis=1).values
Y=heart_disease.target.values

#X
#Y

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=1,test_size=0.2)

#X_train
#Y_test
#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Logistic Regression
heart_disease_model = LogisticRegression()
heart_disease_model.fit(X_train,Y_train)

Y_pred = heart_disease_model.predict(X_test)
#Y_pred

print('Logistic Accuracy Score:',accuracy_score(Y_test, Y_pred)*100,'%')
print('Logistic Precision Score:',precision_score(Y_test, Y_pred)*100,'%')
print('Logistic Recall Score:',recall_score(Y_test, Y_pred)*100,'%')
print('Logistic F1 Score : ', f1_score(Y_test, Y_pred)*100, '%\n')

#Random Forest Classifier
np.random.seed(42)
heart_disease_rf_clf = RandomForestClassifier(n_estimators = 10, max_depth = 1, random_state=(1))
heart_disease_rf_clf.fit(X_train, Y_train)

Rf_Y_pred = heart_disease_rf_clf.predict(X_test)
#Rf_Y_pred

print('Random Forest Accuracy Score:',accuracy_score(Y_test, Rf_Y_pred)*100,'%')
print('Random Forest Precision Score:',precision_score(Y_test, Rf_Y_pred)*100,'%')
print('Random Forest Recall Score:',recall_score(Y_test, Rf_Y_pred)*100,'%')
print('Random Forest F1 Score : ', f1_score(Y_test, Rf_Y_pred)*100, '%\n')

#Support Vector Regressor
np.random.seed(42)
SVC_clf = SVC()
SVC_clf.fit(X_train, Y_train)

SVC_Y_pred = SVC_clf.predict(X_test)
#SVC_Y_pred

print('SVC Accuracy Score:',accuracy_score(Y_test, SVC_Y_pred)*100,'%')
print('SVC Precision Score:',precision_score(Y_test, SVC_Y_pred)*100,'%')
print('SVC Recall Score:',recall_score(Y_test, SVC_Y_pred)*100,'%')
print('SVC F1 Score : ', f1_score(Y_test, SVC_Y_pred)*100, '%\n')

#K Neighbors Classifier
np.random.seed(42)
KNN_clf = KNeighborsClassifier(n_neighbors = 7, weights="distance")
KNN_clf.fit(X_train, Y_train)

KNN_Y_pred = KNN_clf.predict(X_test)
#KNN_Y_pred

print('KNN Accuracy Score:',accuracy_score(Y_test, KNN_Y_pred)*100,'%')
print('KNN Precision Score:',precision_score(Y_test, KNN_Y_pred)*100,'%')
print('KNN Recall Score:',recall_score(Y_test, KNN_Y_pred)*100,'%')
print('KNN F1 Score : ', f1_score(Y_test, KNN_Y_pred)*100, '%')


#Finding best parameters for Random Forest Classifier
#np.random.seed(42)
#for i in range(1,40,1):
  #print(f"With {i*10} estimators:")
  #Rf_clf2=RandomForestClassifier(n_estimators=i*10,max_depth=i,random_state=i).fit(X_train,Y_train)
  #print(f"Accuracy: {Rf_clf2.score(X_test,Y_test)*100:2f}%")
###!!! I used this to find n_estimators, max_depth, and random_state for above Random Forest Classifier 
 
#GridSearch CV
# defining parameter range 
#param_grid = {'C': [0.1, 1,2, 10, 100, 1000],  
              #'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              #'kernel': ['rbf','linear']}  
  
#gs_clf = GridSearchCV(SVC(), param_grid,cv=5, refit = True, verbose = 3) 
  
# fitting the model for grid search 
#gs_clf.fit(X_train, Y_train)

#print(gs_clf.best_params_)
#print(f"\nGrid Search CV Accuracy score:{gs_clf.score(X_test,Y_test)*100}%") 

def evaluation(Y_test,Y_pred):
  acc=accuracy_score(Y_test,Y_pred)
  rcl=recall_score(Y_test,Y_pred)
  f1=f1_score(Y_test,Y_pred)
 

  metric_dict={'accuracy': round(acc,3),
               'recall': round(rcl,3),
               'F1 score': round(f1,3),
               
              }

  return print(metric_dict)

evaluation(Y_test,SVC_Y_pred)

from mlxtend.classifier import StackingCVClassifier
scv=StackingCVClassifier(classifiers=[KNN_clf,heart_disease_rf_clf],meta_classifier= KNN_clf)
scv.fit(X_train,Y_train)
scv_score=scv.score(X_test,Y_test)
scv_Y_pred=scv.predict(X_test)
#print(SVC_score)
evaluation(Y_test,scv_Y_pred)

## Heat map
fig,ax=plt.subplots()
ax=sns.heatmap(confusion_matrix(Y_test,Rf_Y_pred),annot=True,cbar=True);



###########################################################################



###Run in shell
#user_input = input("Enter the values one by one : ")
#user_input = user_input.split(",")

#for i in range(len(user_input)):
    # convert each item to int type
    #user_input[i] = float(user_input[i])

#user_input = np.array(user_input)
#user_input = user_input.reshape(1,-1)
#user_input = scal.transform(user_input)
#final_rf_Y_pred = heart_disease_rf_clf.predict(user_input)
#if(final_rf_Y_pred[0]==0):
 # print("Warning! You have chances of getting a heart disease!")
#else:
  #print("You are healthy and are less likely to get a heart disease!")

######################################################################


pkl.dump(heart_disease_rf_clf,open("final_model.p","wb"))
 
#Load the saved model
model=pkl.load(open("final_model.p","rb"))


#Making web app

# front end elements of the web page 
html_temp = """ 
    <div style ="background-color:#80808b; padding:3px"> 
    <h1 style ="color:red;text-align:center;">Heart Disease Detection App</h1> 
    </div> 
    """
  
from PIL import Image
title_container = st.container()
col1, col2 = st.columns([1,7])
#col2 = st.container()
img = Image.open(r"—Pngtree_ real heart_5953443.png")
with title_container:
    with col1:
        st.image(img, width = 100)
    with col2:
        st.markdown(html_temp, unsafe_allow_html = True)
    
st.write('''
##### by *Hein Htet Hlaing* 
''')


def preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal ):   
 
    
    # Pre-processing user input   
    if sex=="male":
        sex=1 
    else: sex=0
    
    
    if cp=="Typical angina":
        cp=0
    elif cp=="Atypical angina":
        cp=1
    elif cp=="Non-anginal pain":
        cp=2
    elif cp=="Asymptomatic":
        cp=2
    
    if exang=="Yes":
        exang=1
    elif exang=="No":
        exang=0
 
    if fbs=="Yes":
        fbs=1
    elif fbs=="No":
        fbs=0
 
    if slope=="Upsloping: better heart rate with excercise(uncommon)":
        slope=0
    elif slope=="Flatsloping: minimal change(typical healthy heart)":
          slope=1
    elif slope=="Downsloping: signs of unhealthy heart":
        slope=2  
 
    if thal=="fixed defect: used to be defect but ok now":
        thal=6
    elif thal=="reversable defect: no proper blood movement when excercising":
        thal=7
    elif thal=="normal":
        thal=2.31

    if restecg=="Nothing to note":
        restecg=0
    elif restecg=="ST-T Wave abnormality":
        restecg=1
    elif restecg=="Possible or definite left ventricular hypertrophy":
        restecg=2


    user_input=[age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal]
    user_input=np.array(user_input)
    user_input=user_input.reshape(1,-1)
    user_input=scal.fit_transform(user_input)
    prediction = model.predict(user_input)

    return prediction


st.subheader("About App")

st.info("This web app helps you to find out whether you are at a risk of developing a heart disease.")
st.sidebar.info("Enter the required fields and click on the 'Predict' button to check whether you have a healthy heart")

st.write("If you have some problems to complete the required fields,", "[Here](https://share.streamlit.io/mrnightcrypt/heart-disease-assistance/main/assistance.py)")
# Creating User Interface to enter data required to make prediction

age = st.sidebar.selectbox("Age",range(1,121,1))

sex = st.sidebar.radio("Select Gender: ", ('male', 'female'))

cp = st.sidebar.selectbox('Chest Pain Type',("Typical angina","Atypical angina","Non-anginal pain","Asymptomatic")) 

trestbps = st.sidebar.selectbox('Resting Blood Sugar',range(1,500,1))

restecg = st.sidebar.selectbox('Resting Electrocardiographic Results',("Nothing to note","ST-T Wave abnormality","Possible or definite left ventricular hypertrophy"))

chol = st.sidebar.selectbox('Serum Cholestoral in mg/dl',range(1,1000,1))

fbs = st.sidebar.radio("Fasting Blood Sugar higher than 120 mg/dl", ['Yes','No'])

thalach = st.sidebar.selectbox('Maximum Heart Rate Achieved',range(1,300,1))

exang = st.sidebar.selectbox('Exercise Induced Angina',["Yes","No"])

oldpeak = st.sidebar.number_input('Oldpeak')

slope = st.sidebar.selectbox('Heart Rate Slope',("Upsloping: better heart rate with excercise(uncommon)","Flatsloping: minimal change(typical healthy heart)","Downsloping: signs of unhealthy heart"))

ca = st.sidebar.selectbox('Number of Major Vessels Colored by Flourosopy',range(0,5,1))

thal = st.sidebar.selectbox('Thalium Stress Result',range(1,8,1))



#user_input=preprocess(sex,cp,exang, fbs, slope, thal )
pred = preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal)

#Data Displaying
st.write('''
###### Model Dataset used in this program
''')
st.dataframe(heart_disease)

st.write('''
###### \nBar Chart showing counts of *Male* and *Female* (Male = 0, Female = 1)
''')
st.bar_chart(data = sex_count, width = 50, height = 200)

st.write('''
###### \nBar Chart showing counts of *Male* and *Female* with heart disease (Male = 0, Female = 1)
''')
st.bar_chart(data = hd_sex_count, width = 50, height = 200)

#st.pyplot(chart1)

#st.pyplot(chart2)

st.write('''
##### Heatmap
''')
st.pyplot(fig, figsize =(3,4))

st.write('''
### Model Test Accuracy
''')
st.write('''
         83.60655737704919 %
''')

st.write('#### User Interfaces')
st.write('Age : ', age)
st.write('Gender : ', sex)
st.write('Chest Pain Type : ', cp)
st.write('Resting Blood Sugar : ', trestbps)
st.write('Resting Electrocardiographic Results : ', restecg)
st.write('Serum Cholestoral in mg/dl : ', chol)
st.write('Fasting Blood Sugar higher than 120 mg/dl : ', fbs)
st.write('Maximum Heart Rate Achieved : ', thalach)
st.write('Exercise Induced Angina : ', exang)
st.write('Oldpeak : ', oldpeak)
st.write('Heart Rate Slope : ', slope)
st.write('Number of Major Vessels Colored by Flourosopy : ', ca)
st.write('Thalium Stress Result : ', thal)

if st.button("Predict"):    
  if pred[0] == 0:
    st.error('Warning! You have high risk of getting a heart attack!')
  else:
    st.success('You have lower risk of getting a heart disease!')
               
st.info("Caution: This is just a machine learning prediction program and not doctoral advice. Kindly see a doctor if you feel the symptoms persist.")

















