import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import scale


class RandomForest():
    def __init__(self, DATA, START_DATE):
        self.data = DATA
        self.start_date = START_DATE

    def train_test_split(self, df, category):
        df_category = df
        df_category[category] = self.data[category] 
        df_category_dummies = pd.get_dummies(df_category, columns=['Day_name','event_type_1','event_type_2']) 
        y = df_category_dummies[category]
        x = df_category_dummies.drop(category,axis = 1)
        
        x = x.set_index('date')
        
        
        start_forecast_date = self.start_date
  
        x_train = x[:start_forecast_date]
       
        x_test = x[start_forecast_date::]
        y_train = y.loc[:1913]
        y_test = y.loc[1913::]
        st.write("Data Ready After train and test splits")
        return x_train, y_train, x_test, y_test
    
    def plot_results(self, category, rf_random, x_train, y_train, x_test, y_test):

        st.write("Parameters from Random Forest Model for {} category ".format(category))
    
        st.write(rf_random.best_params_)
        # st.write(x_train.shape, y_train.shape,x_test.shape, y_test.shape )
        st.write("Train score",rf_random.score(x_train, y_train))
        st.write("Test score",rf_random.score(x_test, y_test))
        features = pd.Series(rf_random.best_estimator_.feature_importances_, index=x_train.columns)
        st.write('Top 10 features for prediction for {}:'.format(category))
        st.write(features.sort_values(ascending=False)[0:10])
        y_test_pred = pd.Series(rf_random.best_estimator_.predict(x_test).astype('int32'),index=y_test.index)
        # Predict over the train period and combine with test period in order to calculate residuals
        y_train_pred = pd.Series(rf_random.best_estimator_.predict(x_train).astype('int32'),index=y_train.index)

        total_pred = pd.concat([y_train_pred, y_test_pred], axis=0)

        # Plot both the actual data and forecasted data
        fig = plt.figure(figsize=(18,18))
        ax1 = fig.add_subplot(311)
        title = "Actual vs Predicted order_counts for "+category +" :Random Forest"
        ax1.set_title(title)
        plt.plot(y_test, 'b-', label='actual')
        plt.plot(y_test_pred, 'r--', label='predicted')
        plt.xticks(rotation = 45)
        ax1.set_xlabel('Dates')
        ax1.set_ylabel('Order Counts')
        ax1.legend(loc='best')
        st.write(' ')
        st.write('Random Forest Regressor Model for {} categroy - from {}'.format(category,self.start_date))
        st.write('='*50)
        st.write('R2 score on test data: {}'.format(r2_score(y_test, y_test_pred)))
        st.pyplot(fig)

        # Calculate the absolute error in the prediction vs actual number of trips
        abs_error = (y_test - y_test_pred).abs()

        # Calculate the percentage error = absolute_error / actual trips
        percent_error = (abs_error / y_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        st.write(' ')
        st.write('Average absolute error :', abs_error.mean())
        st.write('RMSE score :',rmse)
        st.write('Average percentage error :',percent_error.mean())
