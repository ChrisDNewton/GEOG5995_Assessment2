"""
=======================================================================================================
Assessment 2: Independent Project
GEOG5995 Programming for Social Scientists: Core Skills [Python]
Leeds student number: 201190819

This project produces some descriptive statistics and carries out some tests on the British Social 
Attitudes Survey 2014, available here: https://discover.ukdataservice.ac.uk/catalogue/?sn=7809
=======================================================================================================
"""

"""
=======================================================================================================
Setup - import libraries, set up file structure and import data using Pandas.
=======================================================================================================
"""

#import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
import seaborn as sns
#Set the gridstyle and colour scheme for Seaborn graphs
sns.set(style="whitegrid", color_codes=True) 

#Set up file structure - check if an 'output' folder exists, and create one if not -
if not os.path.isdir ('./output'):
	os.mkdir ('./output')

#read in the data source tab file as a Pandas data frame. Raise exception if absent. 
while True:
    try:
        df = pd.read_table('bsa14_final.tab', low_memory=False)
        break
    except IOError: #If file is absent, alert user
        print("Data not found. Please ensure bsa14_final.tab is saved in project folder.")
        print("This can be downloaded from https://discover.ukdataservice.ac.uk/catalogue/?sn=7809")
        break

"""
=======================================================================================================
Some variables are string objects which need converting to numeric format for analysis. 
Based on code from: 
https://stackoverflow.com/questions/24037507/converting-string-objects-to-int-float-using-pandas
=======================================================================================================
"""
#Convert leftrigh from str object to numeric 
df['leftrigh'] = pd.to_numeric(df['leftrigh'], errors='coerce')

#Convert libauth from str object to numeric 
df['libauth'] = pd.to_numeric(df['leftrigh'], errors='coerce')

#Convert MeatEnv from str object to numeric 
df['MeatEnv'] = pd.to_numeric(df['leftrigh'], errors='coerce')


"""
=======================================================================================================
Define a function for calculating r2. 
from: https://stackoverflow.com/questions/25579227/seaborn-implot-with-equation-and-r2-text
=======================================================================================================
"""

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

"""
=======================================================================================================
Define functions for dealing with missing values. Based on method from 
'Think Stats - Exploratory Data Analysis in Python' ebook (page 9) available at: 
http://greenteapress.com/wp/think-stats-2e/
=======================================================================================================
"""

def CleanData(df):
    na_vals = [9, -1, 0.0] #Specify the missing numbers for this set of variables
    df.deathapp.replace(na_vals, np.nan, inplace=True) #replace missing data in a specific variabel with np.nan (a floating point value that represents non-numerical values)
    df.leftrigh.replace(na_vals, np.nan, inplace=True)
    df.libauth.replace(na_vals, np.nan, inplace=True)
    df.MRsnRel.replace(na_vals, np.nan, inplace=True)
    df.MRsnSavM.replace(na_vals, np.nan, inplace=True)
    df.MRsnSafe.replace(na_vals, np.nan, inplace=True)
    
def CleanLogistic(df):
    na_vals = [9, -1] #Specify the missing numbers for this set of variables
    df.MRsnHlth.replace(na_vals, np.nan, inplace=True)
    df.MRsnEnvt.replace(na_vals, np.nan, inplace=True)
    df.MRsnAnml.replace(na_vals, np.nan, inplace=True)

#Define a function to clean a binary predictor variable 
def CleanDataPred(df):
    na_vals = [8, 9, -1] #Specify the missing numbers 
    df.FdEaNone.replace(na_vals, np.nan, inplace=True) #replace missing data in a specific variabel with np.nan    
    df.MeatEnv.replace(na_vals, np.nan, inplace=True)
    df.Rsex.replace(na_vals, np.nan, inplace=True)
    df.Rsex.replace(1, 'Male', inplace=True) #replace '1' with 'male'
    df.Rsex.replace(2, 'Female', inplace=True) #replace '2' with 'female'

#Define a function for removing '5' from the meathab var so it operates as a scale
def CleanDataMeat(df):
    na_vals = [5, 8, 9, -1] 
    df.MeatHab.replace(na_vals, np.nan, inplace=True)
    
#Run the functions to clean the data
CleanDataMeat(df)
CleanDataPred(df)
CleanLogistic(df)
CleanData(df)


"""
=======================================================================================================
Seaborn data visualisation and analysis.  
Library documentation available at: http://www.seaborn.pydata.org
=======================================================================================================
"""

#Distribution histogram of age
sns.distplot(df.Rage)
plt.title('Distribution histogram of age')
plt.savefig('./output/Distribution histogram of age')
plt.figure()


#Correlation of political stance and attitude towards meat consumption by gender- 
sns.lmplot(x="leftrigh", y="MeatHab", hue="Rsex", data=df, x_estimator=np.mean);
plt.title('Attitude towards meat consumption by political stance/sex')
plt.savefig('./output/meat consumption by political stance and gender') 
plt.figure()

#print regression model details
model = ols("leftrigh ~ MeatHab + Rsex", df).fit()
print(model.summary())

"""
=======================================================================================================
Linear Regression.
Libary documentation available at: https://seaborn.pydata.org/tutorial/regression.html
=======================================================================================================
"""

#Linear regression: Attitude towards meat consumption by political allingment:
sns.jointplot(x="leftrigh", y="MeatHab", data=df, kind="reg", stat_func=r2);
plt.title('Attitude towards meat consumption by political allignment')
plt.subplots_adjust(hspace=0.4) #adjust axis to avoid title overlap
plt.savefig('./output/Pearsons Plot of MeatHab by leftrigh', fontsize=20)
plt.figure()

#Linear regression: Attitude towards meat consumption by libertarianism:
sns.jointplot(x="libauth", y="MeatHab", data=df, kind="reg", stat_func=r2);
plt.title('Attitude towards meat consumption by libertarianism')
plt.subplots_adjust(hspace=0.4) #adjust axis to avoid title overlap
plt.savefig('./output/Pearsons Plot of Meathab by libauth', fontsize=20)
plt.figure()

#Bar chart of attitude towards meat consumption by gender
sns.countplot(x='MeatHab', hue='Rsex', data=df)
plt.title('Attitude towards meat consumption by sex')
plt.savefig('./output/Attitude towards meat consumption by gender')
plt.figure()

"""
=======================================================================================================
Multivariate linear regression. 
Adapted from code available at: http://www.scipy-lectures.org/packages/statistics/index.html
=======================================================================================================
"""

model = ols("MeatHab ~ leftrigh + Rsex + libauth + Rage", df).fit()
print(model.summary())

"""
=======================================================================================================
Logistic Regression.
Libary documentation available at: 
https://seaborn.pydata.org/examples/logistic_regression.html
=======================================================================================================
"""

#age and reduced meat intake for health reasons
sns.lmplot(x="Rage", y="MRsnHlth", hue="Rsex", data=df, y_jitter=.01, logistic=True);
plt.title('Probability of reduced meat consumption for health reasons by age')
plt.savefig('./output/Log regression of reduced meat for health reasons') #having a '/' in the title causes error
plt.figure()

#Bar chart of gender split
sns.countplot(x='MRsnHlth', hue='Rsex', data=df) #Count of people who have reduced intake of meat split by gender.
plt.title('Health reasons for reduced meat intake')
plt.figure()

#age and reduced meat intake for environnmental reasons
sns.lmplot(x="Rage", y="MRsnEnvt", hue="Rsex", data=df, y_jitter=.01, logistic=True);
plt.title('Probability of reduced meat consumption for environmental reasons by age')
plt.savefig('./output/Log regression of reduced meat for env reasons') 
plt.figure()

#Bar chart of gender split
sns.countplot(x='MRsnEnvt', hue='Rsex', data=df) #Count of people who have reduced intake of meat split by gender.
plt.title('Environmental reasons for reduced meat intake')
plt.figure()
#Rotate x-labels - 
#plt.xticks(rotation=-45)

#age and reduced meat intake for animal welfare reasons
sns.lmplot(x="Rage", y="MRsnAnml", hue="Rsex", data=df, y_jitter=.01, logistic=True);
plt.title('Probability of reduced meat consumption for animal welfare reasons by age')
plt.savefig('./output/Log regression of reduced meat for welfare reasons') #having a '/' in the title causes error
plt.figure()

#Bar chart of gender split
sns.countplot(x='MRsnAnml', hue='Rsex', data=df) #Count of people who have reduced intake of meat split by gender.
plt.title('Animal welfare reasons for reduced meat intake')
plt.figure()




