# -*- coding: utf-8 -*-
import pandas as pd
from dateutil.parser import parse
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
os.chdir("C:\\Users\\Sarick\\Documents\\Python Scripts\\Film_Analysis")
#%%

import re
r = re.compile(r'%')  

def cleaner(file):
    df = pd.read_csv(file)
    df = df[~df[df.columns[4]].duplicated()]
    df.columns = range(len(df.columns))
    df.drop([0], axis = 1, inplace = True)
    df.columns = range(len(df.columns))
    
    #Dropping the columns I don't need 
    df.drop([0, 1, 7, 11, 12, 13, 14, 15, 17, 20, 24, 25, 32, 33, 40, 41, 48], axis = 1, inplace = True)
    #GPT = gross per theater
    #TG = total gross
    
    
    columns=['Release Date','Title','Production Budget','Domestic Gross','Worldwide Gross','Genre','Runtime','MPAA','Critic Rating','Weekend 1 Rank','Weekend 1 Gross','Weekend 1 Theaters','Weekend 1 GPT','Weekend 1 TG', 'Weekend 2 Rank','Weekend 2 Gross','Weekend 2 Change','Weekend 2 Theaters','Weekend 2 GPT','Weekend 2 TG', 'Weekend 3 Rank','Weekend 3 Gross','Weekend 3 Change','Weekend 3 Theaters','Weekend 3 GPT','Weekend 3 TG', 'Weekend 4 Rank','Weekend 4 Gross','Weekend 4 Change','Weekend 4 Theaters','Weekend 4 GPT','Weekend 4 TG']
    
    df.columns = columns
    df.dropna(inplace=True, thresh = 26)    
    #df3 = df2[(df2['Weekend 2 Change'].str.contains('%')) | (df2['Weekend 3 Change'].str.contains('%')) | (df2['Weekend 4 Change'].str.contains('%'))]
    
    df = df[(df['Weekend 3 Change'].str.contains('%'))]
    df = df[(df['Weekend 2 Change'].str.contains('%'))]
    for column in columns:
        try:
            df[column] = df[column].map(lambda x: x.replace(',',''))
            df[column] = df[column].map(lambda x: x.replace('$',''))
            df[column] = df[column].map(lambda x: x.replace('G(Rating', 'G'))
            df[column] = df[column].map(lambda x: x.replace('GG', 'G'))
            df[column] = df[column].map(lambda x: x.replace('n/c', '0'))
        except (AttributeError):
            pass    
    
    df.fillna(0, inplace=True)
    
    df['Weekend 2 Theaters'] = df['Weekend 2 Theaters'].replace(',','').astype(int)
    df['Weekend 3 Theaters'] = df['Weekend 3 Theaters'].replace(',','').astype(int)
    #Decided to only look at weekends 1-3 instead of 1-4 due to weekend 4 data missing for a lot more movies
    df.drop(['Weekend 4 Theaters','Weekend 4 Rank','Weekend 4 Gross','Weekend 4 Change','Weekend 4 GPT','Weekend 4 TG'], axis = 1, inplace = True)
    
    df['Runtime'] = df['Runtime'].map(lambda x: int(str(x).split()[0]))    
    df['MPAA'] = df['MPAA'].map(lambda x: str(x).split()[0])
    df['Weekend 1 Rank'] = df['Weekend 1 Rank'].map(lambda x: str(x))
    df['Weekend 2 Change'] = df['Weekend 2 Change'].map(lambda x: int(x.replace('%','')))
    df['Weekend 3 Change'] = df['Weekend 3 Change'].map(lambda x: int(x.replace('%','')))
    df['Release Date'] = df['Release Date'].apply(lambda x: parse(str(x)))
    df.to_csv('thenumbers_5000_scrubbed_v2.csv',index=False)
#%%

def ratings_fixer(file):
    import re
    df = pd.read_csv(file)
    list_of_rating = df['Critic Rating'].tolist()    
    total_rating = [] 
    for rating_string in list_of_rating:
        if rating_string == 'nan' or len(rating_string)==0 or re.findall('\d+%', rating_string)==[]:
            total_rating.append([0,0])
        else:
            list_pair = re.findall('\d+%',rating_string)
            total_rating.append(list_pair)
           
    for i in total_rating:
        if len(i)<2:
            a = total_rating.index(i)
            total_rating.remove(i)
            df.drop(a, axis = 0, inplace=True)
    
            
    critic_rate, audience_rate = zip(*total_rating)
    df['Critic Rate'] = critic_rate
    df['Audience Rate'] = audience_rate
    df.to_csv('thenumbers_5000_scrubbed_v3.csv',index=False)   
    
    df['Critic Rate'] = df['Critic Rate'].map(lambda x: int(str(x).replace('%','')))
    df['Audience Rate'] = df['Audience Rate'].map(lambda x: int(str(x).replace('%','')))
    return df
df = ratings_fixer("")

#%%
def fill_with_mean(file):
    df = pd.read_csv(file)
    #Check Column Types and convert all thecells that say object to numeric
    df.columns.to_series().groupby(df.dtypes).groups
    #Finish converting all columns to numeric
    df[['Production Budget', 'Domestic Gross', 'Worldwide Gross', 'Weekend 1 Rank','Weekend 1 Gross','Weekend 1 Theaters','Weekend 1 GPT','Weekend 1 TG', 'Weekend 2 Rank','Weekend 2 Gross','Weekend 2 Change','Weekend 2 Theaters','Weekend 2 GPT','Weekend 2 TG', 'Weekend 3 Rank','Weekend 3 Gross','Weekend 3 Change','Weekend 3 Theaters','Weekend 3 GPT','Weekend 3 TG', 'Runtime']]= df[['Production Budget', 'Domestic Gross','Worldwide Gross', 'Weekend 1 Rank','Weekend 1 Gross','Weekend 1 Theaters','Weekend 1 GPT','Weekend 1 TG', 'Weekend 2 Rank','Weekend 2 Gross','Weekend 2 Change','Weekend 2 Theaters','Weekend 2 GPT','Weekend 2 TG', 'Weekend 3 Rank','Weekend 3 Gross','Weekend 3 Change','Weekend 3 Theaters','Weekend 3 GPT','Weekend 3 TG', 'Runtime']].apply(pd.to_numeric, errors='coerce')
    #Fill every column with the mean value 
    df = df.replace(0, np.nan)
    df = df.fillna(df.mean())
    df.to_csv("movies_5000_zeros_filledv5")
    return df

#%%
#Run analyses (Random Forest was best so I only kept that)

df = pd.read_csv('movies_5000_zeros_filled.csv')
#Convert to string
df['MPAA'] = df['MPAA'].astype(str)
#Deal with random errors in the ratings section
translator = {'M/PG':'PG-13','R(Rated':'R','GG(Rating':'G','Open':'G','G(Rating':'G','Not':'PG-13'}
df['MPAA']=df['MPAA'].replace(translator)
df_dummies = pd.get_dummies(df['Genre'], drop_first=True) 
df_dummies = pd.concat([df_dummies, pd.get_dummies(df['MPAA'], drop_first=True)], axis=1)
df_regression = df[['Production Budget', 'Weekend 2 Change', 'Weekend 3 Change', 'Weekend 1 Theaters', 'Weekend 2 Theaters', 'Weekend 3 Theaters', 'Weekend 1 GPT', 'Weekend 2 GPT', 'Weekend 3 GPT', 'Critic Rate', 'Audience Rate']]
df_regression = pd.concat([df_regression, df_dummies], axis =1, join_axes=[df_regression.index])


#%%
#Split train and test
X,y = df_regression,df['Worldwide Gross']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics.r2_score(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
model.coef_
model.residuals_         


from scipy.stats import randint as sp_randint
# specify parameters and distributions to sample from
param_dist = {"max_depth": sp_randint(1, 10),
              "max_features": sp_randint(1, 10)}  

clf = RandomForestRegressor(n_estimators=100)
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, cv = 5)
random_search.fit(X_train, y_train)
y_pred2=random_search.predict(X_test)
metrics.r2_score(y_test, y_pred2)

# Build the Regressor to show feature importances based off of the best params shown in the cross validation section
clf = RandomForestRegressor(n_estimators = 100, max_depth = 7, max_features = 11)
clf.fit(X_train, y_train)
clf.feature_importances_

df_dummies_genre = pd.get_dummies(df['Genre'])
#Append df_dummies_genre to each and then regress

rating_selection = ['G', 'PG', 'PG-13', 'R']
def genre_comparison(rating_selection):
    df_Rating = df[df['MPAA']==rating_selection]
    df_Rating= df_Rating[['Production Budget','Worldwide Gross', 'Weekend 2 Change', 'Weekend 3 Change', 'Weekend 1 Theaters', 'Weekend 2 Theaters', 'Weekend 3 Theaters', 'Weekend 1 GPT', 'Weekend 2 GPT', 'Weekend 3 GPT', 'Critic Rate', 'Audience Rate']]
    df_Rating = pd.concat([df_Rating, df_dummies_genre], axis =1, join_axes=[df_Rating.index])
    return df_Rating
    

#%%
#Random forest on MPAA Rating
df_Rating = genre_comparison(rating_selection[0])
genre_comparison(df_Rating)
X,y = df_Rating.drop('Worldwide Gross', axis = 1),df_Rating['Worldwide Gross']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_dist = {"max_depth": sp_randint(1, 10),
              "max_features": sp_randint(1, 10)}  

clf = RandomForestRegressor(n_estimators=100)
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, cv = 5)
random_search.fit(X_train, y_train)
y_pred2=random_search.predict(X_test)
metrics.r2_score(y_test, y_pred2)







