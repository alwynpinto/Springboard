# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:10:39 2020

@author: WorkStation
"""
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

df = pd.read_csv(r"C:\Users\WorkStation\PythonDocs\Sandwich Analytics\dataclean1.csv")
df.columns = ['inv_number', 'store_num', 'description', 'price', 'sold', 'del', 'sales', 'tot_sls', 'unit_cost', 'cost', 'cost_percent', 'margin', 'profit', 'date', 'year', 'month', 'day', 'weekday', 'store_name','store_city', 'store_county', 'store_state', 'store_location','store_drive_through', 'store_near_school', 'annual_rent_estimate']
df.to_csv(r"C:\Users\WorkStation\PythonDocs\Sandwich Analytics\dataclean1.csv" , index = False)

df.columns
sales_byweek = df.groupby('date').sales.sum()
sales_byweek.plot()

df[df.date == '2012-08-08'].groupby('store_num').sales.sum()
# seems that only Store # 24 was operational on this date and hence not a real indicator of stores perf

df[df.date == '2012-04-28'].groupby('store_num').sales.sum()
# seems that only Store # 7 was operational on this date and hence not a real indicator of stores perf

county_sales = df.groupby('store_county').sales.sum()
county_sales.sort_values(ascending = False).plot(kind='bar')


store_sales = df.groupby('store_num').sales.sum()
store_sales.sort_values(ascending = False).plot(kind='bar')


best_seller = df.groupby('description').sales.sum().sort_values(ascending=False).head(10)
best_seller.plot(kind='bar')

df.groupby('month').sales.sum().plot(kind = 'bar')

january.columns
_ = plt.figure(figsize=(20,12))
_ = plt.plot(january.Date , january.sales)
df.store_num.unique()


df[df.store_num == 2].groupby('month').sales.sum().plot()
