# CloudForAI


## Goal of the project
Based on a dataset in Kaggle: 
https://www.kaggle.com/datasets/mojtaba142/hotel-booking/data
We want to create a model that can provide the ideal daily rate in order to maximize room occupation in the hotel.

## Reason
Less cancellations and higher revenue. <br/>
Easier to maintain staff scheduling. <br/>
Simplify inventory management.

## Models used and metrics
Linear Regression <br/>
XGBOOST<br/>
KNearest Neighbors<br/>
Conclusion is that even though KNN has the best metrics, due to it's slow performance, we concluded that XGBOOST would be better suited to our needs. 

RMSE

## Deployment on Streamlit.io
We have an app running about this topic. The app will predict the average daily rate based on the parameters filled in. The feedback will be from our XGB model that had the best results. <br />
https://cloudforai-ujg3jhutjrgqcgeinmta3j.streamlit.app/

## Contributors
The contributors of this project are GitHub users: NineYearsOld, JordyCrthls and yrjavk.
