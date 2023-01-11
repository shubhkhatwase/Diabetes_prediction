# -*- coding: utf-8 -*-
"""
Created on Wen Jan  4 3:45:11 2023

@author: Shubh Khatwase
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
import sklearn
import joblib

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Model training", page_icon="📈")


st.markdown("<h1 style = 'text-align:center; color:skyblue; font-size:45px; '>Diabetes Predicting Model</h1>", unsafe_allow_html=True)

st.markdown('<p style="background-color:silver;padding-left:7px;color:#4b5320;font-size:20px;border-radius:1.5%;text-align:left">About Dataset</p>', unsafe_allow_html=True)

st.markdown('<p style="color:green;font-size:18px;">Context:<p>',unsafe_allow_html=True)
st.write('This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.')
st.markdown('<p style="color:green;font-size:18px;">Content:<p>',unsafe_allow_html=True)
st.write("Attribute Information:")

col1=st.columns(1)
st.write('''Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

01. Pregnancies: Number of times pregnant
02. Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
03. BloodPressure: Diastolic blood pressure (mm Hg)
04. SkinThickness: Triceps skin fold thickness (mm)
05. Insulin: 2-Hour serum insulin (mu U/ml)
06. BMI: Body mass index (weight in kg/(height in m)^2)
07. DiabetesPedigreeFunction: Diabetes pedigree function
08. Age: Age (years)
09. Outcome: Class variable (0 or 1)
Sources:

(a)  Original owners: National Institute of Diabetes and Digestive and
Kidney Diseases
(b)  Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
     Research Center, RMI Group Leader
     Applied Physics Laboratory
     The Johns Hopkins University
     Johns Hopkins Road
     Laurel, MD 20707
     (301) 953-6231
     (c) Date received: 9 May 1990''')

def header(url):
     st.markdown(f'<p style="background-color:pink;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)

header("Diabetes Dataset")

data=pd.read_csv("D:\DashBoard\diabetes.csv")


# Columns

st.write(data)

def head1(url):
     col1.markdown(f'<p style="background-color:skyblue;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)

col1,col2=st.columns(2)
head1("head")
col1.write(data.head())

def tail(url):
     col2.markdown(f'<p style="background-color:skyblue;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)

tail("Tail")
col2.write(data.tail())


def Viz(url):
     st.markdown(f'<p style="background-color:orange;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)
# ========================================================================
Viz('Visualization')


for i in data.columns:
    if data[i].max()>1:
        plt.figure(figsize=(7,7))
        fig,ax=plt.subplots()
        ax.hist(data[i])
        ax.set_title(i)
        ax.set_xlabel(f'Skewness{round(data[i].skew(),2)}\nkurt: {round(data[i].kurt(),2)}')
        # ax.set_xticks([0,1])
        st.pyplot(fig)

    else:
        
        plt.figure(figsize=(10,10))
        fig,ax=plt.subplots()

        st.write(data[i].value_counts())

        ax.bar(data[i].value_counts().keys(),data[i].value_counts().values,)
        ax.set_title(i)
        # ax.set_xlabel(i)
        ax.set_xticks([0,1])
        st.pyplot(fig)

        plt.figure(figsize=(10,10))
        fig,ax=plt.subplots()
#--------------------------------------------------------------------
# ---------------------------------------------------------------------

def desc(url):
     st.markdown(f'<p style="background-color:orange;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)
desc('Descriptive Statistics')




col4,col5=st.columns(2)

def descr(url):
     col4.markdown(f'<p style="background-color:skyblue;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)

descr("Descriptive Statistics")
col4.write(data.describe())

def null(url):
     col5.markdown(f'<p style="background-color:skyblue;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)

null('Null Values')
col5.write(data.isnull().sum())


st.write(data.info())

# =========================================================================

def pre(url):
     st.markdown(f'<p style="background-color:	orange;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)



col6,col7=st.columns(2)

def head2(url):
     col6.markdown(f'<p style="background-color:	skyblue;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)

head2('Head')
col6.write(data.head())

def null2(url):
     col7.markdown(f'<p style="background-color:	skyblue;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)

null2('Null')
col7.write(data.isnull().sum())


def model(url):
     st.markdown(f'<p style="background-color:	orange;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)

model('Model training')

st.code('''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
# Seperating input and output
data['Outcome'].value_counts()
data.groupby('Outcome').mean()
X = data.drop(columns = 'Outcome', axis=1)
Y = data['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train,Y_train)
# Accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy score of the training data: ",training_data_accuracy)

predicts=classifier.predict(X_test)
test_data_accuracy = accuracy_score(predicts, Y_test)

st.write(f"Accuaracy score of the test data :", test_data_accuracy)''')
#=================================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
# Seperating input and output
X = data.drop(columns = 'Outcome', axis=1)
Y = data['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
model= svm.SVC(kernel='linear')
model.fit(X_train,Y_train)
# Accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy score of the training data: ",training_data_accuracy)

predicts=model.predict(X_test)
test_data_accuracy = accuracy_score(predicts, Y_test)
st.write(f"Accuaracy score of the test data :", test_data_accuracy)
#================================================================================


def eval(url):
     st.markdown(f'<p style="background-color:	orange;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)

eval('Evaluating model')

from sklearn.metrics import ConfusionMatrixDisplay,roc_curve,auc,RocCurveDisplay,confusion_matrix


def c8(url):
     st.markdown(f'<p style="background-color:	skyblue;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)

cnf_matrix=confusion_matrix(Y_test,predicts)
x=ConfusionMatrixDisplay(cnf_matrix,display_labels=model.classes_)
x.plot()
c8('Confusion Matrix')
st.pyplot(plt.show())

from statsmodels.api import Logit

x=model_logit=Logit(Y,X).fit().summary()
print(x)
st.subheader("Summary")
st.write(x)


st.write("---------------------------------------------------------------------")

# ==================================================================
def ty(url):
     st.markdown(f'<p style="background-color:silver;color:#4b5320;font-size:24px;border-radius:100%;text-align:center">{url}</p>', unsafe_allow_html=True)


ty('🙂--------------------THANK YOU---------------------🙂')




