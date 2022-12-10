## Prince Nathan
Date: 2022-12-07

#### Goal 
- Predict price of a laptop if features are given

#### What I learnt from the app
- Apple laptops are pricier than the others

#### Dataset
 - Dataset contains a list of laptops, their specifications and price in Indian rupee.

#### Machine Learning
- Linear Regression was used to train model that can predict prices.

#### WebApp
- The webapp is simple and straightforward allowing users to straight away choose the specifications they want and get a prediction for the average price they would have to pay for a laptop with such specifications.

## Yena Hong
Date: 2022-12-07

#### Goal 
- Apply unsupervised learning to cluster the world cities based on their living costs and explore characteristics of each cluster.
- Locate each cluster on the world map

#### What I learnt from the app
- 4 clusters were formed based on living cost. 
- Cluster 0 has low cost of living but high mortgage interest rate.
- Cluster 1 has similar cost of living as Cluster 3 but has twice the salary .
- Cluster 2 only has Singapore and is the most expensive city to buy a car in.
- Cluster 3 has very high cost of living.

#### Dataset
 - The original data set consists of country names, city names, and categories for cost of living. Yena added continent names, latitude, and longitude on the data set to locate each of city on the world map. There was no need to transform the data.

#### Machine Learning
- K-means clustering was used for the project. It grouped cities based on cost of living.
- She found the optimal number of clusters using elbow curve and silhouette score which were visualized using plots and allowed the user to interact and come to the same conclusion as well

#### WebApp
- The app was well designed with tabs for different topics which provided for clarity and better flow while going through it.
- Sliders, buttons etc. were used to allow user to interact with the plots and understand that conclusions drawn by Yena.
- Interactive world map to visualize the different city clusters was interesting after which details of the different clusters were elaborated.

## Kevin Patel
Date: 2022-12-09

#### Goal 
Based on the patient details and type of illness, to predict the length of stay in the hospital.

#### What I learnt from the app
- There is a normal distribution of age where the mean seems to be around 40 years of age.
- Moderate severity cases are most common especially for the people in ages 30 to 50 and the average number of patients in the gynecology department is highest.
- Patients with minor illnesses have to pay more to avail longer stay in the hospital.

#### Dataset
- Hospital data of many cities and patient details are taken along with the length of stay.
- Kevin found 2 columns: bed_grade and city_code_patient that have null values which are of MNAR missingness type.

#### Machine Learning
- K Nearest Neighbor, Decision Tree Classifier, Neural Network and Logistic Classifier were used to train a model for prediction.
- After model is generated, the classification report and confusion matrix is shown which elaborates the strengths and weaknesses of the model.
- The user is also given the ability to predict based on custom input.

#### WebApp
- The app is well sectioned according to the story using the sidebar and the extensive use of emojis make the webapp fun to go through.
- Many streamlit features were used such as sliders, dropdowns etc. that appear depending on which section the user is currently on.
- The user can give custom inputs and the webapp will return a prediction of hospital stay length using the selected model.

## Rishabh Sareen
Date: 2022-12-09

#### Goal 
- To find the customers that are more likely to leave the bank's credit card service so that the bank can cater to them more

#### What I learnt from the app
- Highly educated people and married people are most likely to leave the service.

#### Dataset
 - The original data set consists of details of each customer such as their income, marital status, etc. and average utilization ratio

#### Machine Learning
- Logistic Regression, Random Forest and SVM were used as machine learning models. Data was scaled and transformed for mainly the logistic regression model.
- Metrics such as confusion matrix, classification report were used to score the power of a model and inferences were drawn based on the results.
- User is given the option to predict using the models.

#### WebApp
- The webapp is well designed with a large amount of user control. Information is shown only if the user requests it.
- Rishabh has also added inferences and conclusions drawn from the data which form a story that provides insight what kind of people are more likely to use or not use credit card services.
- Metrics used are visualized well.
