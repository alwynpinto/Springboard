'''
Data collected from a quick-serve sandwich chain over one year 
provide opportunity for market, sociographic, meteorologic, and 
other factors impacting sales. The weekly sales table contains over 
140,000 rows which each represent summary statistics for the sales
of an individual menu item in one store during one week of the year. 
The data were collected from the point-of-sale system of 19 stores. 
Secondary data regarding weather patterns, population, location, 
competition, and crime were gathered and integrated with the original 
data set.
'''


%matplotlib inline
import numpy as np
import pandas as pd
import datetime as dt
import calendar
import csv
import matplotlib.pyplot as plt
import scipy.stats as stats
from IPython import display
from ipywidgets import interact,widgets
import re
import csv

stores = pd.read_csv(r"C:\Users\WorkStation\PythonDocs\Sandwich Analytics\AppWichStoreAttributes.csv")
data = pd.read_csv(r"file:///C:/Users/WorkStation/PythonDocs/Sandwich Analytics/weekly_sales_10stores.csv")


data.columns
stores.columns

data.rename(columns={'Store_num':'Stores_Num'},inplace=True)
data.columns
data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2) + '-' + data['Day'].astype(str).str.zfill(2))
data['Weekday'] = data['Date'].dt.day_name()

data = pd.merge(data,stores,on=['Store_Num'],how='left')
data.drop(['Store_Weather_Station','Store_Competition_Fastfood','Store_Competition_Otherfood','Store_Traveller_Clients','Store_Minority_Clients'],axis=1,inplace=True)
data.columns
data.dtypes
data['Store_Drive_Through'] = data['Store_Drive_Through'].astype(bool)
data['Store_Near_School'] = data['Store_Near_School'].astype(bool)
data.columns
data.to_csv("C:/Users/WorkStation/PythonDocs/Sandwich Analytics/dataclean1.csv",index=False)

data
