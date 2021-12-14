import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class eda:
    
    def __init__(self, DATA_URL):
        self.DATA_URL = DATA_URL

    @st.cache
    def load_data(self):
        data = pd.read_csv(self.DATA_URL)
        return data

    # @st.cache
    def category_eda(self, option, data):
        if option == "Foods":

            st.subheader("Average count for "+option+' each day in a week')
            fig = plt.figure(figsize=(16, 8))
            sns.set(font_scale=1.4)
            sns.set_style("whitegrid")
            ax = sns.barplot(data = data, x = 'Day_name', y = 'FOODS')
            # for container in ax.containers:
            #     ax.bar_label(container, padding=50)
            plt.title('Average counts for FOOD each day in a week')
            st.pyplot(fig)

            st.subheader("Observing Mean count for days in each month")
# observing mean order counts for days in each month
            order_pattern = data.groupby(['month','Day_name']).agg({'FOODS':'mean'}).reset_index().sort_values('month', ascending= True)
            order_pattern = order_pattern.pivot("month", "Day_name", "FOODS")
            fig, ax = plt.subplots(figsize=(16, 6))
            sns.heatmap(order_pattern/10**4, annot=True, linewidths=.5, ax = ax)
            plt.ylabel('Month')
            plt.xlabel('Days')
            plt.title("Mean count : FOODS Category")
            plt.yticks(rotation = 45)
            st.pyplot(fig)

            st.subheader("Observing coorelation respect to order counts for FOODS category")
            # observing correlation of features in preprocessed dataframe with 'order_count'
            fig = plt.figure(figsize=(8, 12))
            sns.set(font_scale=1.4)
            sns.set_style("whitegrid")
            heatmap = sns.heatmap(data.corr()[['FOODS']].sort_values(by='FOODS', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
            heatmap.set_title('Features Correlating with FOOD counts', fontdict={'fontsize':18}, pad=16)
            st.pyplot(fig)  

            st.subheader("Plotting Time Series plot for FOODS Category")
            self.plotSeries(data,["FOODS"])



        if option == "Household":

            st.subheader("Average count for "+option+' each day in a week')
            fig = plt.figure(figsize=(16, 8))
            sns.set(font_scale=1.4)
            sns.set_style("whitegrid")
            ax = sns.barplot(data = data, x = 'Day_name', y = 'HOUSEHOLD')
            # for container in ax.containers:
            #     ax.bar_label(container, padding=50)
            plt.title('Average counts for HOUSEHOLD each day in a week')
            st.pyplot(fig)

            st.subheader("Observing Mean count for days in each month")
            order_pattern = data.groupby(['month','Day_name']).agg({'HOUSEHOLD':'mean'}).reset_index().sort_values('month', ascending= True)
            order_pattern = order_pattern.pivot("month", "Day_name", "HOUSEHOLD")
            fig, ax = plt.subplots(figsize=(16, 6))
            sns.heatmap(order_pattern/10**4, annot=True, linewidths=.5, ax = ax)
            plt.ylabel('Month')
            plt.xlabel('Days')
            plt.title("Mean count : HOUSEHOLD Category")
            plt.yticks(rotation = 45)
            st.pyplot(fig)

            st.subheader("Observing coorelation respect to order counts for HOUSEHOLD category")
            fig = plt.figure(figsize=(8, 12))
            sns.set(font_scale=1.4)
            sns.set_style("whitegrid")
            heatmap = sns.heatmap(data.corr()[['HOUSEHOLD']].sort_values(by='HOUSEHOLD', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
            heatmap.set_title('Features Correlating with HOUSEHOLD counts', fontdict={'fontsize':18}, pad=16)
            st.pyplot(fig)

            st.subheader("Plotting Time Series plot for HOUSEHOLD Category")
            self.plotSeries(data,["HOUSEHOLD"])
        
        
        if option == "Hobbies": 

            st.subheader("Average count for "+option+' each day in a week')
            fig = plt.figure(figsize=(16, 8))
            sns.set(font_scale=1.4)
            sns.set_style("whitegrid")
            ax = sns.barplot(data = data, x = 'Day_name', y = 'HOBBIES')
            # for container in ax.containers:
            #     ax.bar_label(container, padding=50)
            plt.title('Average counts for HOBBIES each day in a week')
            st.pyplot(fig) 
            
            st.subheader("Observing Mean count for days in each month")
            order_pattern = data.groupby(['month','Day_name']).agg({'HOBBIES':'mean'}).reset_index().sort_values('month', ascending= True)
            order_pattern = order_pattern.pivot("month", "Day_name", "HOBBIES")
            fig, ax = plt.subplots(figsize=(16, 6))
            sns.heatmap(order_pattern/10**4, annot=True, linewidths=.5, ax = ax)
            plt.ylabel('Month')
            plt.xlabel('Days')
            plt.title("Mean count : HOBBIES Category")
            plt.yticks(rotation = 45)
            st.pyplot(fig) 

            st.subheader("Observing coorelation respect to order counts for HOBBIES category")
            fig = plt.figure(figsize=(8, 12))
            sns.set(font_scale=1.4)
            sns.set_style("whitegrid")
            heatmap = sns.heatmap(data.corr()[['HOBBIES']].sort_values(by='HOBBIES', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
            heatmap.set_title('Features Correlating with HOBBIES counts', fontdict={'fontsize':18}, pad=16)
            st.pyplot(fig) 

            st.subheader("Plotting Time Series plot for HOBBIES Category")
            self.plotSeries(data,["HOBBIES"])

    def plotSeries(self, inputData,lists=[""]):
        for i in lists:
            fig = go.Figure(data = [
                go.Scatter(y = inputData[i],x = inputData[i].index,name = "Time Series Plot for Hobbies")
            ])
            fig.update_layout(title_text=i)
            st.plotly_chart(fig)