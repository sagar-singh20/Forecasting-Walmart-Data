import streamlit as st
import pandas as pd
import numpy as np
from walmart_eda import eda
import seaborn as sns
import matplotlib.pyplot as plt
from random_forest import RandomForest
import joblib
from lstm_model import lstm_model
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from zipfile import ZipFile

file_name = ['LSTM_Foods.zip','LSTM_Hobbies.zip','LSTM_Household.zip']
for each in file_name:
    with ZipFile(each, 'r') as zipObj:
    # Extract all the contents of zip file in current directory
        zipObj.extractall()



st.image('images/walmart.png', use_column_width = 'always')
START_DATE = '2016-04-25'
DATA_URL = 'Data/final_sales_df.csv'
# st.title('Walmart Sales Forecasting')
option_names = ["EDA", "Modeling", "About"]

output_container = st.empty()

option = st.sidebar.radio("Walmart Sales Forecasting", option_names , key="radio_option")
# st.session_state

explore = eda(DATA_URL)
data = explore.load_data()
rf_results = RandomForest(data, START_DATE )
lstm_results = lstm_model(data, START_DATE)

if option == 'EDA':
   
    st.header('Exploratory Data Analysis')
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    
    print("Dev Working..")
    data = explore.load_data()
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')

    if st.button('Quick Glance at the DATA'):
        st.subheader('Walmart Sales Data')
        st.write(data)

    st.header('EDA for different Categories')
    option = st.selectbox('Select the Category',
                            ('Foods','Household','Hobbies'))

    explore.category_eda(option, data)                        


elif option == "Modeling":
    st.header("Modeling on Walmart Data")
    method = st.radio(
     "Modeling Methods",
     ('Random Forest', 'LSTM'))

    if method == 'Random Forest':
        
        st.write('Results from Random Forest')
        sales_cat_s = pd.read_csv('Data/sales_cat_s.csv')
        st.header('Forecasting Results for different Categories')
        option = st.selectbox('Select the Category',
                            ('Foods','Household','Hobbies'))
        
        if option == "Foods":

            x_train, y_train, x_test, y_test =  rf_results.train_test_split(sales_cat_s, 'FOODS')
            rf_random = joblib.load('models/rf_random_foods.joblib')
            rf_results.plot_results('FOODS', rf_random, x_train, y_train, x_test, y_test)
        
        elif option == "Household":
            x_train, y_train, x_test, y_test =  rf_results.train_test_split(sales_cat_s, 'HOUSEHOLD')
            rf_random_house = joblib.load('models/rf_random_household.joblib')
            rf_results.plot_results('HOUSEHOLD', rf_random_house, x_train, y_train, x_test, y_test)

        elif option == "Hobbies":
            x_train, y_train, x_test, y_test =  rf_results.train_test_split(sales_cat_s, 'HOBBIES')
            rf_random_hobbies = joblib.load('models/rf_random_hobbies.joblib')
            rf_results.plot_results('HOBBIES', rf_random_hobbies, x_train, y_train, x_test, y_test)

    elif method == 'LSTM':
        st.write("Results from LSTM Model")
        sales_cat_s = pd.read_csv('Data/sales_cat_s.csv')
        st.header('Forecasting Results for different Categories')
        option = st.selectbox('Select the Category',
                            ('Foods','Household','Hobbies'))
        
        if option == "Foods":
            res  = lstm_results.hampel_filter_forloop(data["FOODS"], 2)
            st.write("STEP 1 - Outliers removed successfully")

            data["FOODS"] = res
            x_train, y_train, x_test, y_test =  lstm_results.train_test_split(sales_cat_s, data, 'FOODS')
            st.write("STEP 2 - Train and test split successfull")

            x_train_sc = lstm_results.scaled_data(x_train)
            x_test_sc = lstm_results.scaled_data(x_test)
            y_train_sc, scaler_tr = lstm_results.scale_y(y_train)
            y_test_sc, scaler_test= lstm_results.scale_y(y_test)
            st.write("STEP 3 - Data Scaling successful")

            x_train, x_test, y_train, y_test = lstm_results.transform_df_LSTM(x_train_sc, x_test_sc,y_train_sc, y_test_sc)
            st.write("STEP 4 - Fetching result from LSTM MODEL for Foods")
            model = keras.models.load_model('LSTM_Foods')
            # make a prediction

            predicted = lstm_results.predict_lstm(model, x_test,y_test, scaler_test)

            lstm_results.plot_results("FOODS", y_test, predicted, scaler_test)




        if option == "Household":
            res  = lstm_results.hampel_filter_forloop(data["HOUSEHOLD"], 2)
            st.write("STEP 1 - Outliers removed successfully")

            data["HOUSEHOLD"] = res
            sales_cat_s["FOODS"] = data["FOODS"]
            x_train, y_train, x_test, y_test =  lstm_results.train_test_split(sales_cat_s, data, 'HOUSEHOLD')
            st.write("STEP - 2 Train and test split successfull")

            x_train_sc = lstm_results.scaled_data(x_train)
            x_test_sc = lstm_results.scaled_data(x_test)
            y_train_sc, scaler_tr = lstm_results.scale_y(y_train)
            y_test_sc, scaler_test = lstm_results.scale_y(y_test)
            st.write("STEP 3 - Data Scaling successful")

            x_train, x_test, y_train, y_test = lstm_results.transform_df_LSTM(x_train_sc, x_test_sc,y_train_sc, y_test_sc)
            st.write("STEP 4 - Fetching result from LSTM MODEL for Household")
            model = keras.models.load_model('LSTM_Household')
            # make a prediction

            predicted = lstm_results.predict_lstm(model, x_test,y_test, scaler_test)

            lstm_results.plot_results("HOUSEHOLD", y_test, predicted, scaler_test)

            
            

        if option == "Hobbies":
            res  = lstm_results.hampel_filter_forloop(data["HOBBIES"], 2)
            st.write("STEP 1 - Outliers removed successfully")

            data["HOBBIES"] = res
            sales_cat_s["FOODS"] = data["FOODS"]
            x_train, y_train, x_test, y_test =  lstm_results.train_test_split(sales_cat_s, data, 'HOBBIES')
            st.write("STEP 2 - Train and test split successfull")
            x_train_sc = lstm_results.scaled_data(x_train)
            x_test_sc = lstm_results.scaled_data(x_test)
            y_train_sc, scaler_tr = lstm_results.scale_y(y_train)
            y_test_sc, scaler_test = lstm_results.scale_y(y_test)
            st.write("STEP 3 - Data Scaling successful")

            x_train, x_test, y_train, y_test = lstm_results.transform_df_LSTM(x_train_sc, x_test_sc,y_train_sc, y_test_sc)
            st.write("STEP 4 - Fetching result from LSTM MODEL for Hobbies")
            model = keras.models.load_model('LSTM_Hobbies')
            # make a prediction

            predicted = lstm_results.predict_lstm(model, x_test,y_test, scaler_test)

            lstm_results.plot_results("HOBBIES", y_test, predicted, scaler_test)

        
    
else:
    st.subheader("App Created and Designed by:")
    st.image("images/contri.png")
    st.subheader("Context")
    st.write("Ecommerce has been an ever-growing industry with retail revenues projected to grow to 4.9 trillion US dollars in 2021. With this tremendous growth, sales forecasts will help every business understand changing customer demands, manage inventories as per the demands thus reducing the financial risks, and create a pricing strategy that reflects demand. Companies will be able to take strategic steps on their short - term and long - term performances and can decide their decision metrics. This project will present the right methodologies to analyze time-series sales data and predict 28 days ahead point forecasts for the company to take strategic decisions based on the predictions. Additionally, the project’s goal includes making recommendations on inventory management based on the 28 days forecast. Sales forecasting is a very crucial research area with companies heavily investing and proposing advanced Methods like FaceBook’s Neural Prophet, Amazon’s DeepAR Model, Dilated convolutional neural networks. We plan to leverage the traditional time series forecasting methods as well as the modern forecasting methods and analyze the time-series sales data for Walmart.")
    # output_container.write("Modeling on Walmart Data")

    st.subheader("Proposed Plan")
    st.write("Making predictions about the future is called extrapolation in the classical statistical handling of time series data. We plan to address this problem statement by using M5 forecasting competition 2020. This is Walmart’s dataset with information about various products sold in the US into 3 different states California, Texas, and Wisconsin. The dataset involves the unit sales of 3049 products classified in 3 product categories(Food, Household, Hobbies) and 7 product departments across a timespan of 5 years starting from 2011 to 2016. The Data also includes explanatory variables such as sell prices, promotions, days of the week, and special events that typically affect unit sales and could improve forecasting accuracy. For the forecasting and predictions on sales data, we are planning to use 3 fundamental approaches to attain best possible results:")
    st.image("images/methods.png")

    st.subheader("Risks and Mitigations")
    st.write("When it comes to Sales Forecasting, accuracy plays an important role to inform decision making. If a prediction is too optimistic, businesses may overinvest in products and personnel, resulting in squandered funds. Companies may underinvest if the prediction is too low, resulting in a shortage of goods and a poor customer experience. However, high accuracy remains challenging for two reasons: Traditional forecasts struggle to incorporate large amounts of previous data, resulting in the omission of significant past signals that get lost in the noise. Traditional forecasts rarely include related but independent data (Exogenous Variables) that can provide important context (for example, price, holidays/events, stock-outs, marketing promotions, and so on). As a result of this, most forecasts fail to accurately predict the future without the full history and context. To mitigate the above risks, we will be leveraging models like SARIMA, and various Deep Learning and Ensemble based models that can account for exogenous variables and handle both large data and multi step forecasting. For implementation, the potential risks would be:")
    st.image("images/risks.png")

    st.subheader("Model Benchmarks/Results")
    st.image("images/model.png")

    st.subheader("Learning & Outcomes")
    st.image("images/learning.png")

    st.subheader("Thank You!")
    