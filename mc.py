from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt 
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import os
import numpy as np
import altair as alt

app_mode = st.sidebar.selectbox(
    'Select Page', ['Home', 'Grafik', 'Prediction', 'About'])
data = pd.read_csv('Water_Quality.csv')
def Graphic():   
    st.title('Water Quality Prediction Graphic ')
    st.write('<h4 style="text-align: center; font-weight: bold;">Dataset</h4>',
            unsafe_allow_html=True)
    dataku = pd.read_csv('Water_Quality.csv')
    st.dataframe(dataku)
    st.write('<h4 style="font-weight: bold;">Describe from Dataset</h4>',
            unsafe_allow_html=True)
    st.dataframe(dataku.describe())
    
    d=pd.DataFrame(dataku["Potability"].value_counts())

    fig=px.pie(d, values="count", names = ["Not Potable","Potable"], hole=0.4, opacity=0.8, 
            labels={"label":"Potability","count":"Number of Samples"})

    fig.update_layout(title=dict(text="Pie Chart of Potability Feature"))
    fig.update_traces(textposition = "outside", textinfo="percent+label")
    st.plotly_chart(fig)
    
    non_potable = dataku.query("Potability == 0")
    potable = dataku.query("Potability == 1")

    plt.figure(figsize=(15,15))
    for ax, col in enumerate(dataku.columns[:9]):
        plt.subplot(3,3, ax+1)
        plt.title(col)
        sns.kdeplot(x=non_potable[col], label="Non Potable")
        sns.kdeplot(x=potable[col], label="Potable")
        plt.legend()
    plt.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Potable Graph in every element")
    st.pyplot()
    
    x =data[['ph', 'Hardness', 'Solids', 'Organic_carbon']]
    y =data['Potability']

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the model
    model_regresi = DecisionTreeRegressor()
    model_regresi.fit(x_train, y_train)

    # Make predictions
    model_regresi_pred = model_regresi.predict(x_test)


    # Display scatter plot
    st.subheader("Actual Potability vs Predicted Potability")
    fig, ax = plt.subplots()
    ax.scatter(x_test.iloc[:, 0], y_test, label='Actual Potability', color='blue')
    ax.scatter(x_test.iloc[:, 0], model_regresi_pred, label='Predicted Potability', color='red')
    ax.set_xlabel('ph')
    ax.set_ylabel('Potability')
    ax.legend()
    st.pyplot(fig)
    st.write('Graphic ini Menggunakan Scatter Plot, yang menampilkan perbandingan tentang hasil prediksi Potability dengan data Potability Asli')

def Prediksi():
    data = pd.read_csv('Water_Quality.csv')  # Ganti 'dataset_motor.csv' dengan nama dataset Anda
    X = data[['ph', 'Hardness','Solids','Organic_carbon']]
    y = data['Potability']

    # Memisahkan data menjadi data latih (training) dan data uji (testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Membuat model Decision Tree Regressor
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    st.title('Water Quality Prediction')
    ph = st.number_input("PH : ", value=None, placeholder="Type a number...")
    Hardness = st.number_input("Hardness : ", value=None, placeholder="Type a number...")
    Solids = st.number_input("Solids : ", value=None, placeholder="Type a number...")
    Organic_Carbon = st.number_input("Organic Carbon : ", value=None, placeholder="Type a number...")

    if st.button('Prediksi'):
        new_data = pd.DataFrame({'ph': [ph], 'Hardness': [Hardness],'Solids' : [Solids], 'Organic_carbon' : Organic_Carbon})
        predicted_cc = model.predict(new_data)
        
        if predicted_cc[0] == 1.0:
            hasil = "Dapat Dikonsumsi Manusia"
        else :
            hasil = "Tidak Dapat Dikonsumsi Manusia"
        st.markdown(f'<p style="background-color:#4F7942;color:#FFFFFF;"> Air dengan ph : {ph}<br>Hardness : {Hardness}<br>Solids : {Solids}<br>Organic Carbon : {Organic_Carbon}<br>----------------------------------------<br>{hasil}</p>', unsafe_allow_html=True)

def About():
    st.markdown("<br>",unsafe_allow_html=True)
    st.write('<h1>Water Quality Prediction</h1>',unsafe_allow_html=True)
    st.image('image.png')
    with open("deskripsi.txt", "r") as f:
        data = f.read()
    st.write(data)
    
def Home():
    st.write('<h1 style="text-align: center; font-weight: bold;">Water Quality Prediction</h1>',
             unsafe_allow_html=True)
    st.image('image.png', caption='Your Image Caption', use_column_width=True, width=400 )
    st.write('<p style="text-align: center;">Created by :<br>1. Dwi Maharani<br>2. Hanum Tyas Nurani<br>3. Istiqomah Sugiarti</p>',unsafe_allow_html=True)
if app_mode == "Grafik":
    Graphic()
elif app_mode == "Prediction":
    Prediksi()
elif app_mode == "About":
    About()
else:
    Home()