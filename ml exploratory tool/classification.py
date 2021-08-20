# To do :
# - joblib to export pipeline
# - explication joblib et comment utiliser ce model
# - ajouter l'option d'ajouter un csv et de travailler dessus
# - 



import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer 
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

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

def get_pipeline_missing_num(imputer, scaler):
    if imputer == 'None':
        return 'drop'
    if imputer == 'Mean':
        pipeline = make_pipeline(SimpleImputer(strategy='mean', missing_values=np.nan))
    if imputer == 'Median':
        pipeline = make_pipeline(SimpleImputer(strategy='median', missing_values=np.nan))
    if(scaler != 'None'):
        pipeline.steps.append(('scaling', get_scaling(scaler)))
    return pipeline


def get_pipeline_missing_cat(imputer, encoder):
    if imputer == 'None' or encoder == 'None':
        return 'drop'
    if imputer == 'Most frequent value':
        pipeline = make_pipeline(SimpleImputer(strategy='most_frequent', missing_values=np.nan))
    pipeline.steps.append(('encoding', get_encoding(encoder)))
    return pipeline

def get_encoding(encoder):
    if encoder == 'None':
        return 'drop'
    if encoder == 'OneHotEncoder':
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def get_scaling(scaler):
    if scaler == 'None':
        return 'passthrough'
    if scaler == 'Standard scaler':
        return StandardScaler()
    if scaler == 'MinMax scaler':
        return MinMaxScaler()
    if scaler == 'Robust scaler':
        return RobustScaler()

def get_ml_algorithm(algorithm):
    if algorithm == 'Logistic regression':
        return LogisticRegression()
    if algorithm == 'Support vector':
        return SVC()
    if algorithm == 'K nearest neighbors':
        return KNeighborsClassifier()
    if algorithm == 'Random forest':
        return RandomForestClassifier()

def get_fold(algorithm, nb_splits):
    if algorithm == 'Kfold':
        return KFold(n_plits = nb_splits, shuffle=True, random_state = 0)


#configuration of the page
st.set_page_config(layout="wide")

title_spacer1, title, title_spacer_2 = st.beta_columns((.1,1,.1))
with title:
    st.title('Classification exploratory tool')
    st.markdown("""
            This app allows you to test different machine learning 
            algorithms and combinations of preprocessing techniques 
            to classify passengers from the Titanic dataset!
            """)

st.write("")


df = get_data_titanic()

target_selected = 'Survived'

X = df.drop(columns = target_selected)
Y = df[target_selected].values.ravel()

#Sidebar 
#selection box for the different features
st.sidebar.title('Preprocessing')
st.sidebar.subheader('Dropping columns')
missing_value_threshold_selected = st.sidebar.slider('Max missing values in feature (%)', 0,100,30,1)
cols_to_remove = st.sidebar.multiselect('Remove columns', X.columns.to_list())

st.sidebar.subheader('Column transformation')
categorical_imputer_selected = st.sidebar.selectbox('Handling categorical missing values', ['None', 'Most frequent value', 'Delete row'])
numerical_imputer_selected = st.sidebar.selectbox('Handling numerical missing values', ['None', 'Median', 'Mean', 'Delete row'])

encoder_selected = st.sidebar.selectbox('Encoding categorical values', ['None', 'OneHotEncoder'])
scaler_selected = st.sidebar.selectbox('Scaling', ['None', 'Standard scaler', 'MinMax scaler', 'Robust scaler'])
text_encoder_selected = st.sidebar.selectbox('Encoding text values', ['None', 'CountVectorizer', 'TfidfVectorizer'])

st.header('Original dataset')

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.beta_columns((0.02, 1.5, 0.2, 1, 0.02))

with row1_1:
    st.write(df)

with row1_2:
    number_features = len(X.columns)

    #feature with missing values
    drop_cols = cols_to_remove
    for col in X.columns:
        #put the feature in the drop trable if threshold not respected
        if((X[col].isna().sum()/len(X)*100 > missing_value_threshold_selected) & (col not in drop_cols)):
            drop_cols.append(col)
    

    #numerical columns
    num_cols_extracted = [col for col in X.select_dtypes(include='number').columns]
    num_cols = []
    num_cols_missing = []
    cat_cols = []
    cat_cols_missing = []
    for col in num_cols_extracted:
        if(len(X[col].unique()) < 25):
            cat_cols.append(col)
        else:
            num_cols.append(col)
        
    #categorical columns
    obj_cols = [col for col in X.select_dtypes(include=['object']).columns]
    text_cols = []
    text_cols_missing = []
    for col in obj_cols:
        if(len(X[col].unique()) < 25):
            cat_cols.append(col)
        else:
            text_cols.append(col)

    #remove dropped columns
    for element in drop_cols:
        if element in num_cols:
            num_cols.remove(element)
        if element in cat_cols:
            cat_cols.remove(element)
        if element in text_cols:
            text_cols.remove(element)

    #display info on dataset
    st.write('Original size of the dataset', X.shape)
    st.write('Dropping ', round(100*len(drop_cols)/number_features,2), '% of feature for missing values')
    st.write('Numerical columns : ', round(100*len(num_cols)/number_features,2), '%')
    st.write('Categorical columns : ', round(100*len(cat_cols)/number_features,2), '%')
    st.write('Text columns : ', round(100*len(text_cols)/number_features,2), '%')

    st.write('Total : ', round(100*(len(drop_cols)+len(num_cols)+len(cat_cols)+len(text_cols))/number_features,2), '%')
    
    #create new lists for columns with missing elements
    for col in X.columns:
        if (col in num_cols and X[col].isna().sum() > 0):
            num_cols.remove(col)
            num_cols_missing.append(col)
        if (col in cat_cols and X[col].isna().sum() > 0):
            cat_cols.remove(col)
            cat_cols_missing.append(col)
        # if (col in text_cols and X[col].isna().sum() > 0):
        #     text_cols.remove(col)
        #     text_cols_missing.append(col)

    #combine text columns in one new column because countVectorizer does not accept multiple columns
    X['text'] = X[text_cols].astype(str).agg(' '.join, axis=1)
    for cols in text_cols:
        drop_cols.append(cols)
    text_cols = 'text'

st.write('cat_col_missing', cat_cols_missing)

#need to make two preprocessing pipeline too handle the case encoding without imputer...
preprocessing = make_column_transformer(
    (get_pipeline_missing_cat(categorical_imputer_selected, encoder_selected) , cat_cols_missing),
    (get_pipeline_missing_num(numerical_imputer_selected, scaler_selected) , num_cols_missing),

    (get_encoding(encoder_selected), cat_cols),
    (get_encoding(text_encoder_selected), text_cols),
    (get_scaling(scaler_selected), num_cols)
)


st.sidebar.title('Cross validation')
type = st.sidebar.selectbox('Type', ['KFold', 'StratifiedKFold'])
nb_splits = st.sidebar.slider('Number of splits', min_value=3, max_value=20)
folds = get_fold(type, nb_splits)

st.sidebar.title('Model selection')
classifier_list = ['Logistic regression', 'Support vector', 'K nearest neighbors', 'Random forest']
classifier_selected = st.sidebar.selectbox('', classifier_list)



preprocessing_pipeline = Pipeline([
    ('preprocessing' , preprocessing)
])


pipeline = Pipeline([
    ('preprocessing' , preprocessing),
    ('ml', get_ml_algorithm(classifier_selected))
])

cv_score = cross_val_score(pipeline, X, Y, cv=folds)
preprocessing_pipeline.fit(X)
X_preprocessed = preprocessing_pipeline.transform(X)

st.header('Preprocessed dataset')
st.write(X_preprocessed)

st.subheader('Results')
st.write('Accuracy : ', round(cv_score.mean()*100,2), '%')
st.write('Standard deviation : ', round(cv_score.std()*100,2), '%')




