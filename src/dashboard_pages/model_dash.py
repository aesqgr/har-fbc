import sys
print(sys.path)
sys.path.append("..")
sys.path.append("../src")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split



model = load_model('src/model_fld/model.h5')


#build dataframe
columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
df_har = pd.read_csv('./src/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt', header = None, names = columns)
df_har = df_har.dropna()
df_har.shape
df_har['z-axis'] = df_har['z-axis'].str.replace(';', '')
df_har['z-axis'] = df_har['z-axis'].apply(lambda x:float(x))
df = df_har[df_har['timestamp'] != 0]
df = df.sort_values(by = ['user', 'timestamp'], ignore_index=True)

#misc
history = pd.read_csv('./src/model_fld/history_test.csv', header = [0])


def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Data Cleaning & Model architecture')
    st.write('Echemos un vistazo a los datos...')
    st.write("**Original shape:** ", df_har.shape)
    st.write("**Final shape:** ", df.shape)
    st.write("**DataFrame reduction:** ", "{:.0%}".format((df.shape[0]/df_har.shape[0] - 1)))
    st.write(df.head())

    st.write("Veamos como se reparten las muestras por cada actividad")
    sns.set_style("whitegrid")
    st.write(sns.countplot(x = "activity", data = df))
    plt.title("Number of samples by activity")
    st.pyplot(plt.show())

    st.write("Cada actividad tiene una forma característica")
    data36 = df[(df["user"] == 36) & (df["activity"] == "Jogging")][:400]
    columns_select = st.beta_columns(2)
    with columns_select[0]:
        data36 = df[(df["user"] == 36) & (df["activity"] == "Jogging")][:400]
        sns.lineplot(y = "x-axis", x = "timestamp", data = data36)
        sns.lineplot(y = "y-axis", x = "timestamp", data = data36)
        sns.lineplot(y = "z-axis", x = "timestamp", data = data36)
        plt.legend(["x-axis", "y-axis", "z-axis"])
        plt.ylabel("Jogging")
        plt.title("Jogging", fontsize = 15)
        st.pyplot()
        data36 = df[(df["user"] == 36) & (df["activity"] == "Walking")][:400]
        sns.lineplot(y = "x-axis", x = "timestamp", data = data36)
        sns.lineplot(y = "y-axis", x = "timestamp", data = data36)
        sns.lineplot(y = "z-axis", x = "timestamp", data = data36)
        plt.legend(["x-axis", "y-axis", "z-axis"])
        plt.ylabel("Walking")
        plt.title("Walking", fontsize = 15)
        st.pyplot()
        data36 = df[(df["user"] == 36) & (df["activity"] == "Upstairs")][:400]
        sns.lineplot(y = "x-axis", x = "timestamp", data = data36)
        sns.lineplot(y = "y-axis", x = "timestamp", data = data36)
        sns.lineplot(y = "z-axis", x = "timestamp", data = data36)
        plt.legend(["x-axis", "y-axis", "z-axis"])
        plt.ylabel("Upstairs")
        plt.title("Upstairs", fontsize = 15)
        st.pyplot()
    with columns_select[1]:
        data36 = df[(df["user"] == 36) & (df["activity"] == "Downstairs")][:400]
        sns.lineplot(y = "x-axis", x = "timestamp", data = data36)
        sns.lineplot(y = "y-axis", x = "timestamp", data = data36)
        sns.lineplot(y = "z-axis", x = "timestamp", data = data36)
        plt.legend(["x-axis", "y-axis", "z-axis"])
        plt.ylabel("Downstairs")
        plt.title("Downstairs", fontsize = 15)
        st.pyplot()
        data36 = df[(df["user"] == 36) & (df["activity"] == "Standing")][:400]
        sns.lineplot(y = "x-axis", x = "timestamp", data = data36)
        sns.lineplot(y = "y-axis", x = "timestamp", data = data36)
        sns.lineplot(y = "z-axis", x = "timestamp", data = data36)
        plt.legend(["x-axis", "y-axis", "z-axis"])
        plt.ylabel("Standing")
        plt.title("Standing", fontsize = 15)
        st.pyplot()
        data36 = df[(df["user"] == 36) & (df["activity"] == "Sitting")][:400]
        sns.lineplot(y = "x-axis", x = "timestamp", data = data36)
        sns.lineplot(y = "y-axis", x = "timestamp", data = data36)
        sns.lineplot(y = "z-axis", x = "timestamp", data = data36)
        plt.legend(["x-axis", "y-axis", "z-axis"])
        plt.ylabel("Sitting")
        plt.title("Sitting", fontsize = 15)
        st.pyplot()
        
    st.image('src/dashboard_pages/imgs/model_summary.png')

    plt.plot(np.array(history['loss']), "r--", label = "Train loss")
    plt.plot(np.array(history['accuracy']), "g--", label = "Train accuracy")
    plt.plot(np.array(history['val_loss']), "r-", label = "Validation loss")
    plt.plot(np.array(history['val_accuracy']), "g-", label = "Validation accuracy")
    plt.title("Training session's progress over iterations")
    plt.legend(loc='lower left')
    plt.ylabel('Training Progress (Loss/Accuracy)')
    plt.xlabel('Training Epoch')
    plt.ylim(0) 
    st.pyplot()
    
    st.image('src/dashboard_pages/imgs/conf_matrix.png')


    