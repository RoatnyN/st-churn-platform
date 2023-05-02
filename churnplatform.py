import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

df=pd.read_csv("/Users/roatny/Desktop/Masterclass/Churnplatform/data/Customer Churn.csv")
#import plost

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('/Users/roatny/Desktop/Masterclass/Churnplatform/churnplatform.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('CHURN PREDICTION `Dashboard`')

# Sidebar
st.sidebar.subheader('4 Powerful ML Models for Churn Prediction')

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load breast cancer dataset
data = pd.read_csv('/Users/roatny/Desktop/Masterclass/Churnplatform/data/Customer Churn.csv')
X=data.drop('Churn',axis=1)
y=df['Churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the four machine learning models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Define a function to train and evaluate a model
def train_and_evaluate_model(model):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy score
    score = accuracy_score(y_test, y_pred)
    
    return score

# Define a function to assign color based on score
def assign_color(score):
    if score >= 0.9:
        return '#4CBB17'
    elif score >= 0.8:
        return '#6693F5'
    else:
        return '#B200ED'


# Create a selectbox to choose the model
model_name = st.sidebar.selectbox('Select a model', list(models.keys()))

# Train and evaluate the selected model
score = train_and_evaluate_model(models[model_name])

# Assign color based on score
color = assign_color(score)

# Define CSS for the box around the accuracy score
box_style = f'background-color: {color}; color: white; padding: 10px; border-radius: 5px;'

# Display the accuracy score with a colored box
st.sidebar.markdown(f'<div style="{box_style}">Accuracy score for {model_name}: <strong>{score:.3f}</strong></div>', unsafe_allow_html=True)

st.sidebar.markdown('''
---
Created with ❤️ by [RNN](https://www.linkedin.com/in/roatny-nuon).
''')

st.subheader('DATASETS')
st.markdown("**In this study, the dataset that we used to build churn model is collected from an Iranian telecom company. Telecom company’s dataset contains 3 150 rows, with 16 columns and each row representing a customer over a year period.**")

#Virtual of all columns in dataset
all_cl=df.columns
fig, ax = plt.subplots()
sns.countplot(y=all_cl)
st.pyplot(fig)

st.subheader('Why Customer Churn?')
st.markdown('**Understanding more about how 5 Key Features and Churn related:**')
df_new=df.drop(['Complains','Status','Distinct Called Numbers','Age','FN','FP','Tariff Plan','Age Group','Charge  Amount','Call  Failure'],axis=1)
fig = sns.pairplot(df_new, hue="Churn")
st.pyplot(fig)

#References
if st.button("References"):
    st.text('[1] G. Benjamin, O. Yasin, “Customer churn prediction using machine learning,” Blekinge Institute of Technology, June 2021.')
    st.text('[2] A. Kasem Ahmad, “Customer churn prediction in telecom using machine learning in big data platform”, Journal of Big Data, 2019.')
    st.text('[3] Briker, Vitaly; Farrow, Richard; Trevino, William; and Allen, Brent (2019) "Identifying Customer Churn in After-market Operations using Machine Learning Algorithms," SMU Data Science Review: Vol. 2: No. 3, Article 6.')
    st.text('[4] Venkata. P.M, “Customer Churns Prediction Model Based on Machine Learning Techniques: A Systematic Review”, Atlantis press, 2021.')
    st.text('[5] AL-Najjar, D.; Al-Rousan, N.; AL-Najjar, H. Machine Learning to Develop Credit Card Customer Churn Prediction. J. Theor. Appl. Electron. Commer. Res. 2022, 17, 1529–1542. https://doi.org/ 10.3390/jtaer17040077.')
    st.text('[6] Mage, “Machine learning (ML) applications: customer churn prediction”, 2022')
    st.text('[7] S. Rekha, “Performance Assurance Model for Applications on SPARK Platform”, Conference Paper · August 2017.')

    #Prediction