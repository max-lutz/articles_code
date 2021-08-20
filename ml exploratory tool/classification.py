import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer

from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


#Loading the data
def get_data_titanic():
    return pd.read_csv(os.path.join(os.getcwd(), 'titanic.csv'))

def get_imputer(imputer):
    if imputer == 'None':
        return 'drop'
    if imputer == 'Most frequent value':
        return SimpleImputer(strategy='most_frequent', missing_values=np.nan)
    if imputer == 'Mean':
        return SimpleImputer(strategy='mean', missing_values=np.nan)
    if imputer == 'Median':
        return SimpleImputer(strategy='median', missing_values=np.nan)

def get_encoder(encoder):
    if encoder == 'None':
        return 'drop'
    if encoder == 'OneHotEncoder':
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def get_scaler(scaler):
    if scaler == 'None':
        return 'passthrough'
    if scaler == 'Standard scaler':
        return StandardScaler()
    if scaler == 'MinMax scaler':
        return MinMaxScaler()
    if scaler == 'Robust scaler':
        return RobustScaler()

        

def get_pipeline_missing_num(imputer, scaler):
    if imputer == 'None':
        return 'drop'
    pipeline = make_pipeline(get_imputer(imputer))
    pipeline.steps.append(('scaling', get_scaler(scaler)))
    return pipeline


def get_pipeline_missing_cat(imputer, encoder):
    if imputer == 'None' or encoder == 'None':
        return 'drop'
    pipeline = make_pipeline(get_imputer(imputer))
    pipeline.steps.append(('encoding', get_encoder(encoder)))
    return pipeline


def get_ml_algorithm(algorithm):
    if algorithm == 'Logistic regression':
        return LogisticRegression()
    if algorithm == 'Support vector':
        return SVC()
    if algorithm == 'K nearest neighbors':
        return KNeighborsClassifier()
    if algorithm == 'Random forest':
        return RandomForestClassifier()


#configuration of the page
st.set_page_config(layout="wide")


st.title('Classification exploratory tool')
st.markdown("""
            This app allows you to test different machine learning 
            algorithms and combinations of preprocessing techniques 
            to classify passengers from the Titanic dataset!
            """)

df = get_data_titanic()

st.header('Original dataset')
st.write(df)

target_selected = 'Survived'
cat_cols_missing = ['Embarked']
num_cols_missing = ['Age']
cat_cols = ['Pclass', 'SibSp', 'Parch', 'Sex']
num_cols = ['Fare']
drop_cols = ['PassengerId']

X = df.drop(columns = target_selected)
y = df[target_selected].values.ravel()

#Sidebar 
#selection box for the different features
st.sidebar.title('Preprocessing')

st.sidebar.subheader('Column transformation')
categorical_imputer_selected = st.sidebar.selectbox('Handling categorical missing values', ['None', 'Most frequent value'])
numerical_imputer_selected = st.sidebar.selectbox('Handling numerical missing values', ['None', 'Median', 'Mean'])

encoder_selected = st.sidebar.selectbox('Encoding categorical values', ['None', 'OneHotEncoder'])
scaler_selected = st.sidebar.selectbox('Scaling', ['None', 'Standard scaler', 'MinMax scaler', 'Robust scaler'])


#need to make two preprocessing pipeline too handle the case encoding without imputer...
preprocessing = make_column_transformer(
    (get_pipeline_missing_cat(categorical_imputer_selected, encoder_selected) , cat_cols_missing),
    (get_pipeline_missing_num(numerical_imputer_selected, scaler_selected) , num_cols_missing),

    (get_encoder(encoder_selected), cat_cols),
    (get_scaler(scaler_selected), num_cols),
    ("drop" , drop_cols)
)


preprocessing_pipeline = Pipeline([
    ('preprocessing' , preprocessing)
])

preprocessing_pipeline.fit(X)
X_preprocessed = preprocessing_pipeline.transform(X)

st.header('Preprocessed dataset')
st.write(X_preprocessed)


st.sidebar.title('Model selection')
classifier_list = ['Logistic regression', 'Support vector', 'K nearest neighbors', 'Random forest']
classifier_selected = st.sidebar.selectbox('', classifier_list)


pipeline = Pipeline([
    ('preprocessing' , preprocessing),
    ('ml', get_ml_algorithm(classifier_selected))
])

folds = KFold(n_splits = 10, shuffle=True, random_state = 0)
cv_score = cross_val_score(pipeline, X, y, cv=folds)

st.subheader('Results')
st.write('Accuracy : ', round(cv_score.mean()*100,2), '%')
st.write('Standard deviation : ', round(cv_score.std()*100,2), '%')