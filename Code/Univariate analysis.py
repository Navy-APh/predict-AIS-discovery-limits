import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from scipy.stats import percentileofscore

#from fitter import Fitter, get_common_distributions, get_distributions

#_____________________________________________________________________________________________________________________

df1 = pd.read_csv ("Period 1_sanitized_and_processed.csv") # read the csv file with all data
df2 = pd.read_csv ("Period 2_sanitized_and_processed.csv") # read the csv file with all data

# Merge df1 and df2 into df_all
df_all = pd.concat([df1, df2], ignore_index=True)
#Covariance matrices -------------------------------------------------------------------
df_selected_features = df_all.iloc[:, 1:10]

# Assuming you have already created df_selected_features as shown before
scaler = StandardScaler()
# Standardize the columns in df_selected_features
df_selected_features_standardized = pd.DataFrame(scaler.fit_transform(df_selected_features), columns=df_selected_features.columns)

covariance_matrix = df_selected_features_standardized.cov()

ax = plt.axes()
sn.heatmap(covariance_matrix, annot=True, fmt='g')
ax.set_title("Covariance matrix")
plt.show() 


'''
# Plot separate for each period ------------------------------------------------

# Plot line chart for the 'Rings' column for period 1
df1['Numerical Date'] = pd.to_datetime(df_all['Numerical Date'], unit='D', origin='12/30/1899')

plt.figure(figsize=(10, 6))
plt.plot(df1['Numerical Date'], df1['Rings'])
plt.ylabel('Covered areas', fontsize=14)
plt.xticks(rotation=45)

# Format x-axis ticks as dates
date_formatter = mdates.DateFormatter('%Y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_formatter)
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig('PLots/Period 1 variation.png', dpi=300)
plt.show()

# Plot line chart for the 'Rings' column for 240 hours in period 1

plt.figure(figsize=(10, 6))
plt.plot(df1['Numerical Date'][1000:1241], df1['Rings'][1000:1241])
plt.ylabel('Covered areas', fontsize=14)
plt.xticks(rotation=45)

# Format x-axis ticks as dates
date_formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')
plt.gca().xaxis.set_major_formatter(date_formatter)
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig('PLots/10 day variation.png', dpi=300)
plt.show()

# Complement of CDF plot

plt.figure(figsize=(10, 6))
plt.hist(df_all['Rings'], bins=100, cumulative=-1, density=True, histtype='step', color='blue', linewidth=4)
plt.xlabel('Coverend areas - discovery limits', fontsize=14)
plt.ylabel('CDF complement', fontsize=14)

# Increase x and y tick label font sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig('PLots/CDF complement.png', dpi=300)
plt.show()
#----------------------
'''

