import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Iris Prediction App

This app predicts the **Iris** species!
""")

st.sidebar.header('User Input Features')


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        sepal_length = st.sidebar.slider('sepal_length',4.3,7.9,5.4)
        sepal_width = st.sidebar.slider('sepal_width',2.0,4.4,3.4)
        petal_length = st.sidebar.slider('petal_length',1.0,6.9,1.3)
        petal_width = st.sidebar.slider('petal_width',0.1,2.5,0.2)
        data = { 'sepal_length' : sepal_length,
                 'sepal_width'  : sepal_width,
                 'petal_length' : petal_length,
                 'petal_width'  : petal_width,
    }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
Iris = pd.read_csv('Iris.csv')

Iris = Iris.drop(columns=['Id','Species'])


df = input_df

df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')



if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('Iris_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_species = np.array(['setosa','versicolor','virginica'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
