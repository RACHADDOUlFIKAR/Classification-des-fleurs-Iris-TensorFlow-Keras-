# lab12 :Classification des fleurs Iris
# realisé par rachad doulfikar
# Import package

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import streamlit as st

#Step 1 : Dataset
iris= datasets.load_iris()
print(iris.data)
print(iris.target)
print(iris.target_names)
# Step 2 : Model
Model=RandomForestClassifier()
# Step 3 : Train
Model.fit(iris.data,iris.target)
#Step 4 : Test
#prediction=Model.predict([[5.9 ,3.,  5.1 ,1.8]])

#Model deplotment on "streamlit"

st.header("Classification des fleurs Iris")
def user_input():
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 6.0)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 3.0)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)

    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    flower_features = pd.DataFrame(data,index=[0])
    return flower_features


#st.write(iris.data)
#st.write(iris.target_names[prediction])

df = user_input()
st.write(df)
st.subheader("Flower Iris prediction")
prediction=Model.predict(df)
st.write(iris.target_names[prediction])
st.image(iris.target_names[prediction][0] + '.jfif', caption=iris.target_names[prediction][0])
st.image('iris_dataset.png')
