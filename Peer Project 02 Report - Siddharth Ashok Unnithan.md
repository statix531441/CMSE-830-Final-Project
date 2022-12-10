## Yena Hong
Date: 2022-10-26

#### Goal 
Analyze default rate and macroeconomic indicators to make predictions for a company to mitigate risk.

#### What I learnt from the app
- The relations and correlations between different macroeconomic indicators and default rate of companies.
- Three major contributors of default rate is inflation rate, unemployment rate and import price index regardless of whether its during a recession or not.

#### Visualization

- Did the visualizations improve your ability to understand the story/narrative?
> Yes, the graphs used aided the story. We first saw trends of all variables over the years and was able to find graphs that had the same peaks and valleys. Then, we went over the correlation matrices and found that the three variables that had the most impact during recessions and during periods of no recession were the same. The three were inflation rate, unemployment rate and import price index.

- What visualization strategies were used? (e.g., use of gray) did the visualizations improve your ability to understand the goal?
> Grey vertical boxes on the background with text indicating whether there was recession during the time and what kind of recession it was. These boxes brought attention to the variations of graphs during these times.

- Which plot libraries were used, and do you feel these were good choices?  
> Plotly and seaborn were used for the presentation and I feel like they were used appropriately to show trends and correlations between features.

#### Dataset and WebApp
- What was the dataset? were there any issues with the data? did the presenter need to  perform any scaling, transformations or remove missing values? 
> The dataset used in this project is time series from 2002 to 2022, including two economic recessions, the Great Recession in 2008 and the COVID-19 recession in 2020, with the annual default rate and monthly macroeconomic indices. The type of recessions and the occurring time were referred from the National Bureau of Economic Research, and the macroeconomic indices were given by Kaggle. The annual default rates used in this project were from the S&P Global Ratings report.

- Was the app well designed? for example, what were the sliders and drop downs used?  were these good choices?
> The app was well designed with various streamlit features used such as regular dropdowns, multi-select dropdowns, buttons, etc. and they were appropriately chosen to guide the user along the story. 


## Kevin Patel
Date: 2022-10-26

#### Goal 
Based on the patient details and type of illness, to predict the length of stay in the hospital.

#### What I learnt from the app
- There is a normal distribution of age where the mean seems to be around 40 years of age.
- Moderate severity cases are most common especially for the people in ages 30 to 50 and the average number of patients in the gynecology department is highest.
- Patients with minor illnesses have to pay more to avail longer stay in the hospital.

#### Visualization

- Did the visualizations improve your ability to understand the story/narrative?
> Yes, various graphs were used in the business conclusion section that supported Kevin's presentation and inferences.


- What visualization strategies were used? (e.g., use of gray) did the visualizations improve your ability to understand the goal?
> Various graphs such as histograms, pie-charts, box plots, cat plots etc. were used that went along with the presentation to build a narrative.
> In the data-analysis section, we are given access to the dataset and can plot various graphs with different libraries.

- Which plot libraries were used, and do you feel these were good choices?  
> Most frequently visible graphs are either matplotlib graphs or seaborn graphs. These were good choices as they are less cluttered and display precisely the narrative Kevin is trying to build. 
> In certain sections, we can play with different libraries such as hiplot.

#### Dataset and WebApp
- What was the dataset? were there any issues with the data? did the presenter need to  perform any scaling, transformations or remove missing values?  
> Kevin found 2 columns: bed_grade and city_code_patient that have null values which are of MNAR missingness type.

- Was the app well designed? for example, what were the sliders and drop downs used?  were these good choices?
> The app is well sectioned according to the story using the sidebar and the extensive use of emojis make the webapp fun to go through.
> Many streamlit features were used such as sliders, dropdowns etc. that appear depending on which section the user is currently on. 