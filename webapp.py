# Import libraries

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

st.sidebar.write("# Index")

chapter = st.sidebar.radio('',
                    ['Introduction',
                    'Variation of Users',
                    'Effect of Season and Weather',
                    'Effect of Numerical Weather Elements',
                    'Effect of Weekends and Holidays',
                    'EDA Inferences',
                    'Feature Engineering',
                    'Linear Regression',
                    'Neural Network',
                    'Kernel Ridge Regression',
                    'User Input',
                    ])

if chapter == 'Introduction':

    # Introduction

    '''
    # Bike Rental Service
    #### By: Siddharth Ashok Unnithan
    ## Introduction

    ### Goal
    To find patterns that can help in understanding demand variations of bike rental service.

    ### Dataset
    **[Bike Sharing in Washington D.C. Dataset](https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset?select=day.csv)**  
    Rental bikes in 2011 and 2012 with corresponding weather and seasonal information.  
    Features:
    - datetime: date and time in 'yyyy-mm-dd hh:mm:ss' format
    - season:  
        1: Winter  
        2: Spring  
        3: Summer  
        4: Fall
    - holiday : Whether day is holiday or not
    - weekday : Day of the week (0-6: Monday-Sunday)
    - workingday : Whether working day or not
    - weather:   
        1: Clear, Partly cloudy  
        2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist  
        3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds  
        4: Heavy Rain + Hail + Thunderstorm + Mist, Snow + Fog  
    - temp : Temperature in Celsius
    - atemp: Apparent temperature.
    - humidity: Humidity
    - windspeed: Wind speed

    Target:  
    - casual: Number of casual users
    - registered: Number of registered users
    - count: Number of total rental bikes including both casual and registered
    '''

df = pd.read_csv('bike_sharing.csv')
df.head()

try:
    df.insert(1, 'date', df['datetime'].apply(lambda x: x[:10]))
    df.insert(2, 'year', df['datetime'].apply(lambda x: x[:4]).astype('int'))
    df.insert(3, 'month', df['datetime'].apply(lambda x: x[5:7]).astype('int'))
    df.insert(4, 'day', df['datetime'].apply(lambda x: x[8:10]).astype('int'))
    df.insert(5, 'hour', df['datetime'].apply(lambda x: x[11:13]).astype('int'))
except:
    pass

cf = df.groupby(['year','month','day']).agg({'hour': np.max,
                                            'season': np.max,
                                            'holiday': np.max,
                                            'workingday': np.max,
                                            'weather': np.max,
                                            'temp': np.mean,
                                            'atemp': np.mean,
                                            'humidity': np.mean,
                                            'windspeed': np.mean,
                                            'casual': np.sum,
                                            'registered': np.sum,
                                            'count': np.sum}).reset_index()

bydate = cf.groupby(['year', 'month']).agg({'hour': np.max,
                                            'season': np.max,
                                            'holiday': np.max,
                                            'workingday': np.max,
                                            'weather': np.max,
                                            'temp': np.mean,
                                            'atemp': np.mean,
                                            'humidity': np.mean,
                                            'windspeed': np.mean,
                                            'casual': np.sum,
                                            'registered': np.sum,
                                            'count': np.sum})
if chapter == 'Variation of Users':
        
    '## Variation Of Users Over Time'

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

    col = st.selectbox(
        "y-axis",
        ['casual', 'registered', 'count'],
        index=2)

    fig, ax = plt.subplots(figsize=(7,5))
    fig.set_facecolor('#0E1117')

    x = np.arange(len(months))

    y = bydate.loc[2011][col]
    ax.plot(x, y, color='#1B5E8D')
    ax.text(x[-1], y[12], s='2011', color='#1B5E8D', horizontalalignment='left', verticalalignment='center', size=16)

    y = bydate.loc[2012][col]
    ax.plot(x, y, color='#F67B0E')
    ax.text(x[-1], y[12], s='2012', color='#F67B0E', horizontalalignment='left', verticalalignment='center', size=16)

    # Settings

    ax.set_xticks(x, months)

    ax.tick_params(axis='both', colors='#ffffff')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0E1117')

    ax.set_title(col.capitalize(), color='white')
    ax.set_ylabel('Number of users (per month)', color='white')

    st.pyplot(fig)

    '''
    1. Total number of people using the service has gone up since 2011.

    2. This increase is more due to the increase in registered users while the number of casual users remain largely the same.

    3. Middle of the year (March to October) has highest number of users which dips towards the end of the year. (Can this be explained from analysing more data?)

    4. We would expect 2012 to have a similar distribution as 2011. However, 2012 saw a sudden dip between june and july. Why?
    '''
##################################################################################

if chapter == 'Effect of Season and Weather':
    '''
    ## Effect of Season and Weather
    Let's see how seasonal and weather changes affect the number of bikers per hour and other features.

    ### Histograms of season and weather:
    '''
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    fig.set_facecolor('#0E1117')

    for i,col in enumerate(['season','weather']):
        ax = axes[i]
        ax.hist(df[col])
        
        # Settings
        ax.tick_params(axis='both', colors='#ffffff')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#0E1117')
        
        ax.set_title(f"{col.capitalize()} histogram", color='white')

    st.pyplot(fig)

    "As enough data is not available for weather == 4, we can expect metrics that don't make sense down the line."

    '### Seasonal effects:'


    seasons = ['Winter', 'Spring', 'Summer', 'Fall']

    byseason = cf.groupby('season').agg({'hour': np.max,
                                        'holiday': np.sum,
                                        'workingday': np.sum,
                                        'weather': pd.Series.mode,
                                        'temp': np.mean,
                                        'atemp': np.mean,
                                        'humidity': np.mean,
                                        'windspeed': np.mean,
                                        'casual': np.mean,
                                        'registered': np.mean,
                                        'count': np.mean})
    # byseason.insert(8,'avg_temp',byseason['temp']/24/100)
    # byseason.insert(10,'avg_atemp',byseason['atemp']/24/100)

    col = st.selectbox(
        "y axis",
        ['holiday', 'workingday', 'temp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
        index=5)

    # FIGURE

    fig, ax = plt.subplots(figsize=(5,3))
    fig.set_facecolor('#0E1117')
    fig.suptitle("Seasonal Effects On " + col.capitalize(), color='white', size=12)

    ax.bar(seasons, byseason[col], color='grey')
    ax.bar(seasons[byseason[col].argmax()], byseason[col].max(), color='#1F77B4')

    if col == 'holiday' or col == 'workingday':
        ax.set_ylabel(f'{col.capitalize()}s (per month)', color='white')

    else:
        ax.set_ylabel(f'{col.capitalize()} (per day)', color='white')

    # Settings
    ax.tick_params(axis='both', colors='#ffffff')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0E1117')

    st.pyplot(fig)

    '''
    1. Summer is the hottest season and it is by far the most popular season for both casual and registered riders.
    2. Registered users ride on both fall and spring while casual users prefer spring over fall.
    3. Both categories prefer winters least.
    '''

    '''### Weather Effects: 
    '''


    byweather = df.groupby(['weather']).mean()
    weathers = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Rain']

    col = st.selectbox(
        "y-axis",
        ['casual', 'registered', 'count'],
        index=0,
        key = 'weatherstuff')

    # FIGURE

    fig, ax = plt.subplots(figsize=(5,3))
    fig.set_facecolor('#0E1117')
    fig.suptitle("Weather Effects On " + col.capitalize(), color='white', size=12)

    ax.bar(weathers, byweather[col], color='grey')
    ax.bar(weathers[byweather[col].argmax()], byweather[col].max(), color='#1F77B4')

    ax.set_ylabel(f'{col.capitalize()} (per hour)', color='white')

    # Settings
    ax.tick_params(axis='both', colors='#ffffff')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0E1117')

    st.pyplot(fig)

    '''
    1. The preference of casual users is clear followed by slightly cloudy, then light rain or snow and they rarely ride during extreme weather conditions.
    2. Registered users have similar preferences except that they seem to enjoy riding during extreme weather conditions. This doesn't reflect the masochistic nature of registered users and only shows that we lack data points for bad weather.
    '''


if chapter == 'Effect of Numerical Weather Elements':

    '''
    ## Effect of Temperature, Apparent Temperature, Windspeed and Humidity:

    First lets look at how temp and atemp are distributed.
    '''

    # GRAPH

    fig, axes = plt.subplots(1,2,figsize=(5,3))
    fig.set_facecolor('#0E1117')
    ax = axes[0]
    ax3 = axes[1]
    # Plot temp and atemp with dropdown
    ax.violinplot(df['temp'])
    ax3.violinplot(df['atemp'])
    # Settings
    ax.tick_params(axis='both', colors='#ffffff')
    ax.xaxis.set_ticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0E1117')

    ax3.tick_params(axis='both', colors='#ffffff')
    ax3.xaxis.set_ticks([])
    ax3.yaxis.set_ticks([])
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['right'].set_color('white')
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.set_facecolor('#0E1117')

    ax2 = ax3.twinx()
    ax2.tick_params('both', colors='white')
    ax2.spines['right'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_ylim((df['temp'].min()*9/5)+32,(df['temp'].max()*9/5)+32)

    # ax4 = ax.twinx()
    # ax4.tick_params('both', colors='white')
    # ax4.spines['right'].set_color('white')
    # ax4.spines['top'].set_visible(False)
    # ax4.spines['bottom'].set_visible(False)
    # ax4.spines['left'].set_visible(False)

    # Titles and labels
    ax.set_title('Temperature', color='white')
    ax3.set_title('Apparent Temp', color='white')
    ax.set_ylabel("Celsius", color='white', rotation=0, ha='right')
    ax2.set_ylabel("Fahrenheit", color='white', rotation=0, ha='left')

    st.pyplot(fig)

    '''
    1. We can see the distribution of temperatures of Washington.

    2. Is temp or atemp related to casual, registered or count?

    3. We also see that the apparent temperature has a slightly different distribution than the actual temperature. (Where does this difference come from?)

    ### How do they vary over time?
    '''

    col = st.selectbox(
        "Select column",
        ['temp', 'atemp', 'humidity', 'windspeed'],
        index=0)

    check = st.checkbox("Lines on temp", value = False)

    fig, ax = plt.subplots(figsize=(7,5))
    fig.set_facecolor('#0E1117')

    x = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

    # Settings
    ax.tick_params(axis='both', colors='#ffffff')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0E1117')

    # Title
    ax.set_title(col.capitalize(), color='white')

    y = bydate.loc[2011][col]
    ax.plot(x, y, color='#1B5E8D')
    ax.text('Dec', y[12], s='2011', color='#1B5E8D', horizontalalignment='left', verticalalignment='center', size=16)
    # ax.vlines('Mar', 0, y[3], color='#1B5E8D', ls=":")
    # ax.hlines(y[3], 'Jan', 'Mar', color='#1B5E8D', ls=":")
    # ax.vlines('Oct', 0, y[10], color='#1B5E8D', ls=":")
    # ax.hlines(y[10], 'Jan', 'Oct', color='#1B5E8D', ls=":")

    y = bydate.loc[2012][col]
    ax.plot(x, y, color='#F67B0E')
    ax.text('Dec', y[12], s='2012', color='#F67B0E', horizontalalignment='left', verticalalignment='center', size=16)
    # ax.vlines('Mar', 0, y[3], color='#F67B0E', ls=":")
    # ax.hlines(y[3], 'Jan', 'Mar', color='#F67B0E', ls=":")
    # ax.vlines('Oct', 0, y[10], color='#F67B0E', ls=":")
    # ax.hlines(y[10], 'Jan', 'Oct', color='#F67B0E', ls=":")
    if col == 'temp' and check:
        ax.hlines(18, 'Jan', 'Dec', color='grey', ls=':')
        ax.hlines(30, 'Jan', 'Dec', color='grey', ls=':')

    st.pyplot(fig)

    '''
    1. We can see that between june and july 2012 had a spike in temperatures which could explain the dip in number of bikers during this time.
    2. Note that the months between March and October have above 15C.
    3. Humidity and Windspeed doesn't share the same kind of graph as casual, registered or count.

    Lets see the relation between temperatures and counts in action.
    '''

    fig, ax = plt.subplots()
    fig.set_facecolor('#0E1117')

    # If y == 'casual' then different textbox positions

    x = st.selectbox(
        "Select x",
        ['atemp', 'temp', 'windspeed', 'humidity'],
        index=1)

    y = st.selectbox(
        "Select y",
        ['casual', 'registered', 'count'],
        index=1)

    sns.regplot(data=bydate.loc[2011], x=x, y=y, lowess=True, ax=ax, label='2011')
    # ax.text(bydate.loc[2011]['atemp'].max()+0.5, bydate.loc[2011][y].max(), '2011', color='#1B5E8D', ha='left', va='center', size=16)

    sns.regplot(data=bydate.loc[2012], x=x, y=y, lowess=True, ax=ax, label='2012')
    # ax.text(bydate.loc[2012]['atemp'].max()+0.5, bydate.loc[2012][y].max(), '2012', color='#F67B0E', ha='left', va='center', size=16)

    # Settings
    ax.tick_params(axis='both', colors='#ffffff')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0E1117')

    # Title
    ax.set_title(f"{x.capitalize()} vs {y.capitalize()}", color='white')

    ax.set_xlabel(x.capitalize(), color='white')
    ax.set_ylabel(y.capitalize() + " (per month)", color='white')
    ax.legend()

    st.pyplot(fig)

    '''
    1. In the temperature vs count graph, the Slope gradually reduces. 
    2. Above 15C is where where the magnitude of the slope starts decreasing which corresponds to the months in between March and October.
    '''
    # ### Is apparent temperature really required?
    # We can discard apparent temperature if we can find atemp as a function of temp, windspeed and humidity.  

    # First, lets plot atemp-temp against windspeed and humidity
    # '''

    # y = bydate['atemp'] - bydate['temp']

    # fig, axes = plt.subplots(2, 1, figsize=(10,15))

    # fig.set_facecolor('#0E1117')

    # for i, col in enumerate(['humidity', 'windspeed']):
        
    #     ax = axes[i]
        
    #     x = bydate[col]
    #     ax.plot(x, y, 'o', color='#1B5E8D')

    #     # Settings
    #     ax.tick_params(axis='both', colors='#ffffff')
    #     ax.spines['bottom'].set_color('white')
    #     ax.spines['left'].set_color('white')
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.set_facecolor('#0E1117')
        
    #     # Title
    #     ax.set_title(f"{col.capitalize()} vs atemp-temp", color='white')
    #     ax.set_xlabel(f"{col.capitalize()}", color='white')
    #     ax.set_ylabel(f"atemp-temp", color='white')
        
    # st.pyplot(fig)

    # '''
    # We see a positive correlation with humidty and negative correlation with windspeed.  
    # Lets try a bubble chart.
    # '''

    # s = bydate['atemp'] - bydate['temp']
    # s = 50 + 950*((s - s.min())/(s.max()-s.min()))

    # fig, ax = plt.subplots(figsize=(10,6))
    # fig.set_facecolor("#0E1117")

    # x = bydate['windspeed']
    # y = bydate['humidity']
    # ax.scatter(x, y, s = s, alpha=0.5)

    # # Settings
    # ax.tick_params(axis='both', colors='#ffffff')
    # ax.spines['bottom'].set_color('white')
    # ax.spines['left'].set_color('white')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_facecolor('#0E1117')

    # # Title
    # ax.set_title("Humidity vs Windspeed (size = atemp-temp)", color='white')
    # ax.set_xlabel("Windspeed", color='white')
    # ax.set_ylabel("Humidity", color='white')

    # st.pyplot(fig)

    # '''
    # 1. Bigger bubbles are concentrated on the top left of the graph and the size decreases as it goes to the bottom right.  
    # 2. We can make the interpretation that high humidity and low windspeed along with temperature can give us an estimate of the apparent temperature. However, from this graph we can note that the "LINEAR" relation isn't strong enough to discard these features.
    # '''

if chapter == 'Effect of Weekends and Holidays':

    '## Effect of Weekends and Holidays'


    byhol = cf.groupby(['workingday', 'holiday']).mean()

    col1 = 'casual'
    col2 = 'registered'

    fig, ax = plt.subplots(figsize=(10,5))
    fig.set_facecolor("#0E1117")

    x = ['Weekend', 'Holiday', 'Working']
    xaxis = np.arange(len(x))

    y11 = byhol.loc[0,0][col1]
    y12 = byhol.loc[0,1][col1]
    y13 = byhol.loc[1,0][col1]

    y1 = [y11,y12,y13]

    y21 = byhol.loc[0,0][col2]
    y22 = byhol.loc[0,1][col2]
    y23 = byhol.loc[1,0][col2]

    y1 = [y11,y12,y13]
    y2 = [y21,y22,y23]

    ax.bar(xaxis-0.2,y1,0.4,label='Casual')
    ax.bar(xaxis+0.2,y2,0.4,label='Registered')

    # Settings
    ax.xaxis.set_ticks(xaxis, x)
    ax.tick_params(axis='both', colors='#ffffff')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0E1117')
    ax.legend()

    # Title
    ax.set_title("Effect of working days and holidays on users", color='white')
    ax.set_ylabel("Number of users (per day)", color='white')

    st.pyplot(fig)

    '''
    1. Casual users rent bikes mostly during weekends and holidays and least during working days.  
    2. This is in contrast with registered users who use the service most during working days - most likely as work/school commute and relatively lesser on holidays and weekends.
    '''

if chapter == 'EDA Inferences':

    '''
    ## EDA Inferences

    ### Summary

    1. Summer is the most popular month for casual and registered users.
    2. Many registered users rent bikes during working days, most likely as commute to work while casual users are more active during holidays and weekends.
    3. Clear and sunny skies with temperatures between 15C and 30C are preferred most by both casual and registered users.
    4. Not enough data is available to understand tendencies of users during bad weather conditions

    ### Utilizing Inferences

    1. We can introduce fluctuating prices by predicting months and days of high demand based on weather conditions for increased profit.
    2. Prices for registered users can be adjusted based on overall usage.
    3. One-time payment to register over fluctuating prices can be a potential marketing strategy to incentivize casual users to register.
    4. Can suggest biking as an healthier, cheaper and environmentally safer alternative to go to work to increase activity of casual users during working days.
    
    '''

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import mean_squared_error, r2_score

cf = df.groupby(['year','month','day']).agg({'season': pd.Series.mode,
                                            'holiday': pd.Series.mode,
                                            'workingday': pd.Series.mode,
                                            'weather': pd.Series.mode,
                                            'temp': np.mean,
                                            'atemp': np.mean,
                                            'humidity': np.mean,
                                            'windspeed': np.mean,
                                            'casual': np.mean,
                                            'registered': np.mean,
                                            'count': np.mean,})

# Defaults

categorical = ['season', 'holiday', 'workingday', 'weather']
numerical = ['temp', 'atemp', 'humidity', 'windspeed']
target = ['casual']

def f(L):
    try:
        return L[0]
    except:
        return L

for cat in categorical:
    cf[cat] = cf[cat].apply(f).astype('int64')


X = cf[categorical + numerical]
y = cf[target]


def transform(X, ohe = OneHotEncoder(sparse=False), sc = StandardScaler(), fit=False):
    if fit:
        X_transformed = np.hstack((ohe.fit_transform(X[categorical]), sc.fit_transform(X[numerical])))
    else:
        X_transformed = np.hstack((ohe.transform(X[categorical]), sc.transform(X[numerical])))
    return X_transformed, ohe, sc

if chapter == 'Feature Engineering':

    f'''
    ## Feature Engineering 

    X = {list(X.columns)}

    ### Feature Selection

    '''

    # Plot
    fig = plt.figure(figsize=(2,2))
    fig.set_facecolor("#0E1117")
    
    ax = plt.subplot(1,1,1)
    ax.imshow(np.abs(X.corr()), cmap='Blues')

    # Settings
    ax.xaxis.set_ticks(range(X.columns.size), X.columns)
    ax.yaxis.set_ticks(range(X.columns.size), X.columns)
    ax.yaxis.tick_right()
    ax.xaxis.tick_top()
    ax.tick_params(axis='both', colors='#ffffff', which='major', labelsize=5)
    ax.tick_params(axis='x', rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor('#0E1117')

    st.pyplot(fig)

    '''
    1. As temp and atemp have very high correlation, we can safely remove atemp.
    2. As we discussed earlier, we should not train on the weather column as we don't have enough data points for different weather conditions
    '''

    categorical = st.multiselect(
        'Choose the categorical columns: ',
        ['season', 'holiday', 'workingday', 'weather'],
        ['season', 'holiday', 'workingday'])

    numerical = st.multiselect(
        'Choose the numerical columns: ',
        ['temp', 'atemp', 'humidity', 'windspeed'],
        ['temp', 'humidity', 'windspeed']
    )

    target = st.selectbox(
    'Choose the target: ',
    ('casual', 'registered', 'count'))

    '''
    ### Train-Test Split

    '''
    X = cf[categorical + numerical]
    y = cf[target]

    test_size = st.slider('', 0.01, 1.0, 0.15, 0.01)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    f'''
    X_train: {X_train.shape}  
    X_test: {X_test.shape}  
    y_train: {y_train.shape}  
    y_test: {y_test.shape}  
    '''
    
    

    f'''
    ## Data Preprocessing

    1.  {categorical} are categorical data so we can use one-hot encoding.
    1.  {numerical} are numerical data so we standardize them using standard scaler.

    '''

    X_train_transformed, ohe, sc = transform(X_train, fit=True)
    X_test_transformed , _, _ = transform(X_test, ohe=ohe, sc=sc)

    st.session_state['stuff'] = categorical, numerical, target, ohe, sc

    f'''
    After the transformation we get something like this:
    '''
    X_train_transformed

    st.session_state['data'] = X_train_transformed, X_test_transformed, y_train, y_test

if chapter == 'Linear Regression':

    '''
    ## Linear Regression
    '''
    try:
        X_train_transformed, X_test_transformed, y_train, y_test = st.session_state['data']

        model = LinearRegression()

        model.fit(X_train_transformed, y_train)

        st.session_state["Linear Regression"] = model

        prediction = model.predict(X_train_transformed)
        train_err = mean_squared_error(y_train, prediction, squared=False)
        
        f'''
        ### Prediction on Train set:
        MSE: {train_err:.5}  
        R2 Score: {r2_score(y_train, prediction):.5}
        '''

        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1,2,1)
        ax.plot(y_train, prediction, 'o')
        ax.set_title("Prediction on X_train", color='white')
        ax.set_xlabel('y_train', color='white')
        ax.set_ylabel('prediction', color='white')

        # Settings
        fig.set_facecolor("#0E1117")
        ax.tick_params(axis='both', colors='#ffffff')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#0E1117')

        prediction = model.predict(X_test_transformed)
        test_err = mean_squared_error(y_test, prediction, squared=False)

        f'''
        ### Prediction on Test set:
        MSE: {test_err:.5}  
        R2 Score: {r2_score(y_test, prediction):.5}
        '''

        ax = fig.add_subplot(1,2,2)
        ax.plot(y_test, prediction, 'o')
        ax.set_title("Prediction on X_test", color='white')
        ax.set_xlabel('y_test', color='white')

        # Settings
        fig.set_facecolor("#0E1117")
        ax.tick_params(axis='both', colors='#ffffff')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#0E1117')

        st.pyplot(fig)
    except:
        '''#### Click on Feature Engineering tab to modify and form the dataset'''





if chapter == 'Neural Network':
    
    '''
    ## Neural Network
    '''
    try:
        X_train_transformed, X_test_transformed, y_train, y_test = st.session_state['data']

        solver = st.selectbox(
            'Select Solver: ',
            ('adam', 'sgd', 'lbfgs')
            )

        activation = st.selectbox(
            'Select activation function: ',
            ('relu', 'tanh', 'logistic', 'identity')
        )

        n_layers = st.slider('Number of hidden layers: ',
            1, 5, 2, 1)

        hidden_layer_sizes = [1]*n_layers

        for layer in range(n_layers):
            hidden_layer_sizes[layer] = st.slider(f"Number of neurons in layer {layer+1}",
                                                    1, 20, max(7*(2-layer), 5), 1)

        model = MLPRegressor(alpha=1e-4, 
                            activation=activation,
                            solver=solver,
                            hidden_layer_sizes=hidden_layer_sizes,
                            max_iter=10000)

        model.fit(X_train_transformed, y_train)

        st.session_state["Neural Network"] = model

        prediction = model.predict(X_train_transformed)
        train_err = mean_squared_error(y_train, prediction, squared=False)
        
        f'''
        ### Prediction on Train set:
        MSE: {train_err:.5}  
        R2 Score: {r2_score(y_train, prediction):.5}
        '''

        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1,2,1)
        ax.plot(y_train, prediction, 'o')
        ax.set_title("Prediction on X_train", color='white')
        ax.set_xlabel('y_train', color='white')
        ax.set_ylabel('prediction', color='white')

        # Settings
        fig.set_facecolor("#0E1117")
        ax.tick_params(axis='both', colors='#ffffff')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#0E1117')

        prediction = model.predict(X_test_transformed)
        test_err = mean_squared_error(y_test, prediction, squared=False)

        f'''
        ### Prediction on Test set:
        MSE: {test_err:.5}  
        R2 Score: {r2_score(y_test, prediction):.5}
        '''

        ax = fig.add_subplot(1,2,2)
        ax.plot(y_test, prediction, 'o')
        ax.set_title("Prediction on X_test", color='white')
        ax.set_xlabel('y_test', color='white')

        # Settings
        fig.set_facecolor("#0E1117")
        ax.tick_params(axis='both', colors='#ffffff')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#0E1117')

        st.pyplot(fig)
    except:
        '''#### Click on Feature Engineering tab to modify and form the dataset'''


if chapter == 'Kernel Ridge Regression':

    '''
    ## Kernel Ridge Regression
    '''
    try:
        X_train_transformed, X_test_transformed, y_train, y_test = st.session_state['data']

        kernel = st.selectbox(
            'Select Kernel: ',
                ['rbf','polynomial','cosine','sigmoid','linear']
            )

        alpha_num = st.slider('Alpha Number: ',
            1, 999, 1, 1)


        alpha_exp = st.slider('Alpha Exponent: ',
            -10, 1, -7, 1)

        gamma_num = st.slider('Gamma Number: ',
            1, 999, 999, 1)


        gamma_exp = st.slider('Gamma Exponent: ',
            -10, 1, -6, 1)


        model = KernelRidge(kernel=kernel,
                        alpha=alpha_num*(10**alpha_exp),
                        gamma=gamma_num*(10**gamma_exp)
        )

        model.fit(X_train_transformed, y_train)

        st.session_state['Kernel Ridge Regression'] = model

        prediction = model.predict(X_train_transformed)
        train_err = mean_squared_error(y_train, prediction, squared=False)
        
        f'''
        ### Prediction on Train set:
        MSE: {train_err:.5}  
        R2 Score: {r2_score(y_train, prediction):.5}
        '''

        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1,2,1)
        ax.plot(y_train, prediction, 'o')
        ax.set_title("Prediction on X_train", color='white')
        ax.set_xlabel('y_train', color='white')
        ax.set_ylabel('prediction', color='white')

        # Settings
        fig.set_facecolor("#0E1117")
        ax.tick_params(axis='both', colors='#ffffff')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#0E1117')

        prediction = model.predict(X_test_transformed)
        test_err = mean_squared_error(y_test, prediction, squared=False)

        f'''
        ### Prediction on Test set:
        MSE: {test_err:.5}  
        R2 Score: {r2_score(y_test, prediction):.5}
        '''

        ax = fig.add_subplot(1,2,2)
        ax.plot(y_test, prediction, 'o')
        ax.set_title("Prediction on X_test", color='white')
        ax.set_xlabel('y_test', color='white')

        # Settings
        fig.set_facecolor("#0E1117")
        ax.tick_params(axis='both', colors='#ffffff')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#0E1117')

        st.pyplot(fig)
    except:
        '''#### Click on Feature Engineering tab to modify and form the dataset'''



if chapter=="User Input":

    f'''
    ## Make Predictions Live
    '''
    try: 
        '''
        The best model results arise from using the Kernal Ridge Regressor with the given default values.  
        However, feel free to choose the model you'd like to make the prediction with:
        '''
        model_name = st.selectbox(
            'Select Model: ',
            ( 'Kernel Ridge Regression', 'Neural Network', 'Linear Regression')
            )

        model = st.session_state[model_name]
        categorical, numerical, target, ohe, sc = st.session_state['stuff']

        weats = ['Cloudy', 'Light Rain', 'Heavy Rain']
        seas = ['Spring', 'Summer', 'Fall', 'Winter']
        boo = ['No', 'Yes']

        df = {}

        for cat in categorical:
            if cat == 'season':
                thing = seas.index(st.selectbox("Season:", seas))+1

            if cat == 'holiday':
                thing = boo.index(st.selectbox("Holiday:", boo))

            if cat == 'workingday':
                thing = boo.index(st.selectbox("Working Day:", boo))

            if cat=='weather':
                thing = weats.index(st.selectbox("Weather:", weats))+1

            df[cat] = [thing]

        for num in numerical:
            thing = st.slider(num.capitalize(), cf[num].min(), cf[num].max(), 0.1)
            df[num] = [thing]

        X = pd.DataFrame(df)
        
        '### Data:'
        st.write(X)

        X_trans, _, _ = transform(X, ohe, sc)

        '### Transformed Data:'
        st.write(X_trans)
        try:
            f'''
            ### Prediction:
            
            We estimate about `{max(0, model.predict(X_trans)[0]):.0f} {target}` users per hour.

            '''
        except:
            f'''
            #### --- Train the model, then try again ---
            '''

        '''
        ### Conclusion:

        The model has learnt how to predict on different features accurately.  
        This can be seen by varying the features above and seeing how the predicted count reacts to it.  

        1. As temperature goes up, the count goes up but beyond a certain point it starts going down.
        2. Holiday and working days have a significant impact on the prediction as we can see the count almost double on holidays and non-working days.
        3. The model trains poorly and makes bad predictions on the weather column if included. This supports the point in EDA Inference - Summary.
        '''
    except:
        f'''#### Click on {model_name} tab to train the model.'''

