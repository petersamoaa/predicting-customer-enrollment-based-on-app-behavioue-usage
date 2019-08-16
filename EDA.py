# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:12:13 2019

@author: Peter Samoaa
"""
#### Importing Libraries ####

import pandas as pd
from dateutil import parser # to parse datetime field 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('appdata10.csv')


#### EDA ####


dataset.head(10) # Viewing the Data
dataset.describe() # Distribution of Numerical Variables, Also to see the data in a more robust way. 

# First set of Feature cleaning
# in the describe function which used for statistical info for numerical data, hour column doesn't appear, 
# because it's a string feature, so we will set it to int 
dataset["hour"] = dataset.hour.str.slice(1, 3).astype(int)

### Plotting
# Get the numerical features and care feature as new dataframe 
dataset2 = dataset.copy().drop(columns = ['user', 'screen_list', 'enrolled_date',
                                           'first_open', 'enrolled'])
dataset2.head()

## Histograms
# To know the distribution of data
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
# because we will plot differnt figures in one figure we will iterate 
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()# cleans up everything 
#    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])

    vals = np.size(dataset2.iloc[:, i - 1].unique()) # bins is the number of recangle in hist which is our case all the values of the column
    
    plt.hist(dataset2.iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.savefig('app_data_hist.jpg')

## Correlation with Response Variable
dataset2.corrwith(dataset.enrolled).plot.bar(figsize=(20,10),
                  title = 'Correlation with Reposnse variable',
                  fontsize = 15, rot = 45, # xaxis is roltated 45, so we don't turn around to read the vertical label
                  grid = True)


## Correlation Matrix
# that help us in model building because we don't want any feature to be dependant on each other 
# becasuse based on the assupmtions of ML model, the features should be independant
sn.set(style="white", font_scale=2)

# Compute the correlation matrix
corr = dataset2.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True # becase the matrix is symetrics so the lower of diagonal is same of above so we need one of them

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


 
#### Feature Engineering ####


# Formatting Date Columns
dataset.dtypes
dataset["first_open"] = [parser.parse(row_date) for row_date in dataset["first_open"]]
dataset["enrolled_date"] = [parser.parse(row_date) if isinstance(row_date, str) else row_date for row_date in dataset["enrolled_date"]]
dataset.dtypes

# Selecting Time For Response
dataset["difference"] = (dataset.enrolled_date-dataset.first_open).astype('timedelta64[h]') # differnce in hours, that's why we used time delta
response_hist = plt.hist(dataset["difference"].dropna(), color='#3F5D7D')
plt.title('Distribution of Time-Since-Screen-Reached')
plt.show()
# As we can see that the highest distribution is between 0 & 500, but the highest distribution in those 500 could be in 100, or even less 
# That's why we define a range between 0 & 100 and we will discover that the highest value is 
# between 0 and 25 
plt.hist(dataset["difference"].dropna(), color='#3F5D7D', range = [0, 100])
plt.title('Distribution of Time-Since-Screen-Reached')
plt.show()

# based on the distribution before we will take the first 48 hours (2 days)
# we changed every response variable to be 0 for difference higher that 48
dataset.loc[dataset.difference > 48, 'enrolled'] = 0
dataset = dataset.drop(columns=['enrolled_date', 'difference', 'first_open'])

## Formatting the screen_list Field

# Load Top Screens
top_screens = pd.read_csv('top_screens.csv').top_screens.values
top_screens

# Mapping Screens to Fields
# here we want to add column for most pupolar screen, then another columns as count for the rest.
dataset["screen_list"] = dataset.screen_list.astype(str) + ',' # this command will create comma as many screen, so we will use the comma for counting


for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int) # Here we create column for each top screen and see if it appear in screen_list of not. return boolean (0 OR 1)
    dataset['screen_list'] = dataset.screen_list.str.replace(sc+",", "") # remove the top screen from screen_list by replace it with empty string. 

dataset['Other'] = dataset.screen_list.str.count(",") # howmany left overscreen do we have by counting ","
dataset = dataset.drop(columns=['screen_list']) # we don't need the column screen_list any more.

# Funnels : Group of screen that belong to the same set
# there are many screens which are correlated to each other and we don't need such screens because that cause a problem for our model. 
# so we group all correlated screen into one funl to become one columns of how many screen it contains, remove the correlated 
# saving_screen in define based on business expert. 
savings_screens = ["Saving1",
                    "Saving2",
                    "Saving2Amount",
                    "Saving4",
                    "Saving5",
                    "Saving6",
                    "Saving7",
                    "Saving8",
                    "Saving9",
                    "Saving10"]
dataset["SavingCount"] = dataset[savings_screens].sum(axis=1) # count all the columns under the saving
dataset = dataset.drop(columns=savings_screens) # remove all columns included in saving screen.

# repeat the same step for the other funnel
cm_screens = ["Credit1",
               "Credit2",
               "Credit3",
               "Credit3Container",
               "Credit3Dashboard"]
dataset["CMCount"] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)

cc_screens = ["CC1",
                "CC1Category",
                "CC3"]
dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)

loan_screens = ["Loan",
               "Loan2",
               "Loan3",
               "Loan4"]
dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)

#### Saving Results ####
dataset.head()
dataset.describe()
dataset.columns

dataset.to_csv('new_appdata10.csv', index = False)

