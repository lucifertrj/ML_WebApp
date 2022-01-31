import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
path = os.path.join("assets","img1.png")

from sklearn.model_selection import train_test_split

#import machine learning models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

#metrics to check the trained ml model
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import mean_squared_error

import model_selection_heart_disease,model_selection_titanic, model_selection_hr, model_selection_hospital

def features():
    params = dict()
    if clf_name == "Logistic Regression":
        slider_lr_c = st.sidebar.slider("c",min_value=0.2,max_value=1.0,step=0.1)
        params['C'] = slider_lr_c
        slider_lr_max_iter = st.sidebar.slider("max_iter",min_value=100,max_value=1500,step=100)
        params['max_iter'] = slider_lr_max_iter

    elif clf_name == "Decision Tree Classifier":
        slider_dc = st.sidebar.slider("max_depth",min_value=3,max_value=15,step=1)
        params['max_depth'] = slider_dc
        slider_dc_nodes = st.sidebar.slider("max_leaf_nodes",min_value=10,max_value=50,step=5)
        params['max_leaf_nodes'] = slider_dc_nodes

    elif clf_name == "Naive Bayes":
        slider_nb_alpha = st.sidebar.slider("alpha",min_value=0.0,max_value=1.0,step=0.1)
        params['alpha'] = slider_nb_alpha

    elif clf_name == "KNN":
        slider_k_value = st.sidebar.slider("n_neighbors",min_value=3,max_value=25,step=2)
        params['n_neighbors'] = slider_k_value

    else:
        slider_rf_n = st.sidebar.slider("n_estimators",min_value=100,max_value=1300,step=100)
        params['n_estimators'] = slider_rf_n
        slider_dc = st.sidebar.slider("max_depth",min_value=3,max_value=15,step=1)
        params['max_depth'] = slider_dc
    return params


#header of the web application
st.title("Machine Learning+EDA Application")
st.markdown("### *Analysis // Visulaize // Predict*")

#sidebar with drop down options
dataset = st.sidebar.selectbox(label="Choose Dataset:",options=['Hospital Survival','Heart Disease','Titanic Survival','HR Job'])
clf_name = st.sidebar.selectbox(label="Choose Classifier ML Model:",options=['Decision Tree Classifier','KNN','Logistic Regression','Random Forest Classifier','Naive Bayes'])
checkbox_feature = st.sidebar.checkbox("Enable Feature Scaling")

#display the option which is active in the
st.markdown("### **Choose ML Model and Dataset From Sidebar**")
st.markdown("*** Dataset Choosen:  *** ``{}``".format(dataset))
st.markdown("***Predicting Using***:  ``{}`` Model".format(clf_name))
if checkbox_feature:
    st.markdown("***Feature Scaling:*** `Feature Scaling Enabled`")
else:
    st.markdown("***Feature Scaling:*** `Feature Scaling Disabled`")

st.markdown("### EDA- Exploratory Data Analysis:")
st.image(path,caption="Anime Vyuh-Tarun Jain")

re_format = dataset.lower().replace(' ','_')+'.csv'
read_dataset = pd.read_csv("./dataset/{}".format(re_format))

st.markdown("### ``Dataset Infomation``")
df = pd.DataFrame(read_dataset)
st.write(df.head(5))

st.markdown("### ``Dataset Infomation``")
st.write(df.describe())
fig,axes = plt.subplots()
st.markdown("""### `Empty Dataset Detection:`""")

for col in df.columns:
    percent = (df[col].isnull().sum()*100)/len(df)
    if percent>0:
        st.write("{} has {}% empty data".format(col,percent))

sb.heatmap(df.isnull())
st.pyplot(fig)

st.markdown("""### `Analysis what you will be predicting`""")
target_prediction_visualize = df.iloc[:,-1].values.tolist()
favour = target_prediction_visualize.count(1)
not_in_favour = target_prediction_visualize.count(0)

fig1,axes1 = plt.subplots()
if dataset == "Hospital Survival":
    sb.countplot(x="Survived_1_year",data=df)
    X = model_selection_hospital.features(checkbox_feature)
elif dataset == "Titanic Survival":
    sb.countplot(x="Survived",data=df)
    X = model_selection_titanic.features(checkbox_feature)
elif dataset == "Heart Disease":
    X = model_selection_heart_disease.features(checkbox_feature)
    sb.countplot(x="target",data=df)
else:
    X = model_selection_hr.features(checkbox_feature)
    sb.countplot(x="target",data=df)
st.pyplot(fig1)

fig2,axes2 = plt.subplots()
plt.pie([favour,not_in_favour],labels=[1,0],autopct="%0.2f%%",colors=["purple","blue"],explode=[0.01,0.02])
st.pyplot(fig2)

fig3,axes3 = plt.subplots()
sb.heatmap(df.corr(),annot=True,linewidths=1)
st.pyplot(fig3)

test_case = st.sidebar.slider("test_size_to_fit_model",min_value=0.15,max_value=0.35,step=0.02)

model_features = features()

st.markdown("`Cross Validation`")

Y = df.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=test_case)

def predict_ml(model):
    model = model.fit(x_train,y_train)
    predict = model.predict(x_test)
    st.markdown("Accuracy Score: `{}`".format(accuracy_score(predict,y_test)))
    st.markdown("Mean Absolute Error: `{}`".format(mean_absolute_error(predict,y_test)))
    st.markdown("Mean Squared Error: `{}`".format(mean_squared_error(predict,y_test)))
    conf_matrix = confusion_matrix(y_test,predict)
    st.markdown("**`Confusion Matrix`**")
    fig10, ax = plt.subplots()
    sb.heatmap(conf_matrix/np.sum(conf_matrix), annot=True,fmt='.2%', cmap='CMRmap')
    st.pyplot(fig10)

st.header("Move the slider to see the variation in model")
def main():
    if clf_name=="Decision Tree Classifier":
        model=DecisionTreeClassifier(max_depth=model_features['max_depth'],max_leaf_nodes=model_features['max_leaf_nodes'])
        predict_ml(model)

    elif clf_name=="Logistic Regression":
        model=LogisticRegression(C=model_features['C'],max_iter=model_features['max_iter'])
        predict_ml(model)

    elif clf_name=="KNN":
        model=KNeighborsClassifier(n_neighbors=model_features['n_neighbors'])
        predict_ml(model)

    elif clf_name=="Random Forest Classifier":
        model=RandomForestClassifier(max_depth=model_features['max_depth'],n_estimators=model_features['n_estimators'])
        predict_ml(model)

    else:
        model = MultinomialNB(alpha=model_features['alpha'])
        predict_ml(model)

if __name__ == '__main__':
    main()