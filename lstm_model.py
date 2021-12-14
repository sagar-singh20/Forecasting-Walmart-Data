import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


class lstm_model():
    def __init__(self, DATA, START_DATE):
        self.data = DATA
        self.start_date = START_DATE

    def hampel_filter_forloop(self,input_series, window_size, n_sigmas=3):
        n = len(input_series)
        new_series = input_series.copy()
        k = 1.4826 # scale factor for Gaussian distribution  
        indices = []
        # possibly use np.nanmedian 
        for i in range((window_size),(n - window_size)):
            x0 = np.median(input_series[(i - window_size):(i + window_size)])
            S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
            if (np.abs(input_series[i] - x0) > n_sigmas * S0):
                new_series[i] = x0
                indices.append(i)
        return new_series

    def train_test_split(self, df, sales_cat, category):
        df_category = df
        df_category[category] = sales_cat[category] 
        df_category_dummies = pd.get_dummies(df_category, columns=['Day_name','event_type_1','event_type_2']) 
        y = df_category_dummies[category]
        x = df_category_dummies.drop(category,axis = 1)
        # st.write(x.columns)
        x = x.set_index('date') 
        start_forecast_date = self.start_date
        x_train = x[:start_forecast_date]
        x_test = x[start_forecast_date::]
        y_train = y.loc[:1913]
        y_test = y.loc[1913::]
        return x_train, y_train, x_test, y_test

    def scaled_data(self,df):
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_df = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)
        return scaled_df
    
    def scale_y(self, df):
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler = scaler.fit(np.array(df).reshape(-1,1))
        new_df = scaler.transform(np.array(df).reshape(-1,1))
        return new_df, scaler

    def transform_df_LSTM(self, X_train, X_test, Y_train, Y_test):
        X_train_arr = np.array(X_train)
        X_test_arr = np.array(X_test)
        
        train_X = X_train_arr.reshape((X_train_arr.shape[0], 1, X_train_arr.shape[1]))
        test_X = X_test_arr.reshape((X_test_arr.shape[0], 1, X_test_arr.shape[1]))
        train_Y = np.array(Y_train).reshape(-1,1)
        test_Y = np.array(Y_test).reshape(-1,1)
        return train_X, test_X, train_Y, test_Y

    def predict_lstm(self, model, x_test,y_test, scaler):
        y_predicted = model.predict(x_test, verbose=0)
        rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
        # st.write('Test RMSE: %.3f' % rmse)
        
        # scaler.fit_transform()
        rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(y_predicted)))
        st.write('Average Percetage Error: %.3f' % rmse)
        return y_predicted



    def plot_results(self, category, test_Y, y_predicted, scaler):

        fig = plt.figure(figsize=(18,18))
        ax1 = fig.add_subplot(311)
        title = "Actual vs Predicted order_counts for "+category +" : LSTM Model"
        ax1.set_title(title)
        act = scaler.inverse_transform(test_Y)
        
        pred = scaler.inverse_transform(y_predicted)
        # st.write(pred)
        plt.plot(act, 'b-', label='actual')
        plt.plot(pred, 'r--', label='predicted')
        plt.xticks(rotation = 45)
        ax1.set_xlabel('Dates')
        ax1.set_ylabel('Differenced Order Counts')
        ax1.legend(loc='best')
        st.pyplot(fig)
        st.write(' ')
        st.write('LSTM Model - from :',self.start_date)
        st.write('='*50)
        st.write('R2 score on test data:',r2_score(test_Y, y_predicted))

        # Calculate the absolute error in the prediction vs actual number of trips
        abs_error = (test_Y - y_predicted)
        # Calculate the percentage error = absolute_error / actual trips
        percent_error = (abs_error / test_Y)

        rmse = np.sqrt(mean_squared_error(test_Y, y_predicted))

        # st.write(' ')
        # st.write('Average absolute error:',abs_error.mean())
        # st.write('RMSE:',rmse)
        # st.write('Average percentage error :', percent_error.mean())

        # Calculate the absolute error in the prediction vs actual number of trips
        abs_error = abs(test_Y - y_predicted)

        # Calculate the percentage error = absolute_error / actual trips
        percent_error = (abs_error / test_Y)

        rmse = np.sqrt(mean_squared_error(test_Y, y_predicted))

        st.write(' ')
        st.write('Average absolute error :', abs_error.mean())
        st.write('RMSE score :',rmse)
        st.write('Average percentage error :',percent_error.mean())
