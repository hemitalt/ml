import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('homework.csv')
new_df = df.drop('income', axis='columns')
income = df.income

plt.xlabel('year')
plt.ylabel('income(USD$)')
plt.title('Income per captia in Canada')
plt.scatter(df.year, df.income)

st.write("""
### Canada's Income Per Captia Predicter using ML and Linear Regression
""")
inputbox =  st.number_input('What year would you like me to predict?',0)


if inputbox == 0.00:
  st.write('Please add a year')

else:
  reg = linear_model.LinearRegression()
  reg.fit(new_df.values,income)
  prediction = reg.predict([[inputbox]])
  string = " ".join(str(x) for x in prediction)
  st.write("""
  USD
  """)
  st.write("$" + string)
