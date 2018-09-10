import pandas as pd
import numpy as np
import matplotlib as plt
#%matplotlib inline

df = pd.read_csv(r"C:\Users\VRMachine\Documents\1._Projects\Python\Load Prediction\training.csv") #Reading the dataset in a dataframe using Pandas

df.apply(lambda x: sum(x.isnull()),axis=0) 

df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 

print("Press Enter to continue ...")
input()