import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import xgboost
import streamlit as st


st.write("""
         #Sales Model""")

# batch data load functie
data = pd.read_csv("hotel_booking.csv")

data.head()

data.info()

data.isna().sum().sort_values(ascending=False)

data.drop(['email', 'credit_card', 'phone-number', 'name'], axis=1, inplace=True)
data['arrival_date_month'] = pd.to_datetime(data.arrival_date_month, format='%B').dt.month
data['reservation_status_date'] = data['reservation_status_date'].astype('datetime64[ns]')

data.drop(['agent', 'company'], axis=1, inplace=True)

data.hist(column='children')

data['children'].fillna(data['children'].median(), inplace=True)
data['children'] = data['children'].astype(int)

mode_country = data['country'].mode()[0]
data['country'] = data['country'].fillna(mode_country)

def factorize_columns(data, columns_to_factorize):
    encoded_mappings = {}

    for column in columns_to_factorize:
        codes, unique_values = pd.factorize(data[column])
        data[column] = codes
        encoded_mappings[column] = {'codes': codes, 'unique_values': unique_values}

    return data, encoded_mappings

def replace_undefined_with_mode(data, column_name):
    mode_value = data[column_name].mode()[0]  # Get the mode (most frequent value)

    # Replace "undefined" values with the mode
    data[column_name] = data[column_name].replace('', mode_value)

    return data

data.isna().sum().sort_values(ascending=False)

data, mappings = factorize_columns(data,
                                   ['hotel', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type',
                                    'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status'])

for column, mapping in mappings.items():
    print("Unique values:", mapping['unique_values'])

fig = plt.figure(figsize=(20, 5))
plt.boxplot(data['adr'], vert=False)
plt.show()

data.drop(data['adr'].idxmax(), inplace=True)

cancelled_percentage = data["is_canceled"].value_counts(normalize=True)
data['total_revenues'] = data['adr'] * (data['stays_in_weekend_nights'] + data['stays_in_week_nights'])
print(cancelled_percentage)

data.head()

plt.figure(figsize=(5, 4))
plt.title("Cancellation Count")
is_canceled_counts = data["is_canceled"].value_counts()

bars = plt.bar(["Not canceled", "Canceled"], is_canceled_counts)
for bar, count in zip(bars, is_canceled_counts):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.1, str(count), fontsize=10)

plt.figure(figsize=(6, 2))
ax1 = sns.countplot(x="hotel", hue="is_canceled", data=data)
legend_labels, _ = ax1.get_legend_handles_labels()
ax1.legend(bbox_to_anchor=(1, 1))
plt.title("Cancellations per hotel", size=20)
plt.xlabel("Hotel")
plt.ylabel("number of reservations")
plt.legend(["not canceled", "canceled"])
plt.show()

cancelled_data = data[data['is_canceled'] == 1]
top_10_countries_canceled = cancelled_data['country'].value_counts()[:10]
plt.figure(figsize=(6, 6))
plt.title("Top 10 countries with cancellations in % ")
plt.pie(top_10_countries_canceled, autopct="%.2f", labels=top_10_countries_canceled.index)
plt.show()

lost_revenues = cancelled_data['total_revenues'].sum()
print(lost_revenues)

top_10_countries_canceled_revenues = cancelled_data.groupby('country')['total_revenues'].sum()[:10]
plt.figure(figsize=(6, 6))
plt.title("Top 10 countries revenue lost due to cancellation in % ")
plt.pie(top_10_countries_canceled_revenues, autopct="%.2f", labels=top_10_countries_canceled_revenues.index)
plt.show()

current_revenues = data[data['is_canceled'] == 0]['total_revenues'].sum()
print(current_revenues)

plt.figure(figsize=(8, 4))
ax1 = sns.countplot(x='arrival_date_month', data=data, hue='is_canceled')
legend_labels, _ = ax1.get_legend_handles_labels()
ax1.legend(bbox_to_anchor=[1, 1])
plt.title("Cancellations per month", size=20)
plt.xlabel('month')
plt.ylabel('number of reservations')
plt.legend(['not cancelled', 'cancelled'])
plt.show()

data[data['is_canceled'] == 0]['is_repeated_guest'].sum()

data['total_stay_in_nights'] = data['stays_in_week_nights'] + data['stays_in_weekend_nights']

data[data['is_canceled'] == 0]

plt.figure(figsize=(6, 4))
sns.lineplot(x='arrival_date_month', y='adr', hue='hotel', data=data)
plt.ylabel("Average daily price")
plt.xlabel("Months")
p = plt.xticks(rotation=30)
plt.title("Average Daily rate by month")

data.drop(['country', 'total_revenues', 'total_stay_in_nights'], axis=1, inplace=True)

def RMSE(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    return np.sqrt(mse)

y = data[['adr']]
X = data.drop(['adr', 'reservation_status_date'], axis=1)

Xtrain, Xrest, ytrain, yrest = train_test_split(X, y, test_size=0.2)
Xval, Xtest, yval, ytest = train_test_split(Xrest, yrest, test_size=0.5)

lr = LinearRegression()
lr.fit(Xtrain, ytrain)

columns = Xtrain.columns.values.tolist()
print(columns)
coefs = lr.coef_.ravel().tolist()
print(coefs)

plt.figure(figsize=(10, 6))
plt.barh(columns, coefs)
plt.xlabel("Feature importance")
plt.ylabel("Features")

X.drop(["is_repeated_guest"], axis=1, inplace=True)

Xtrain, Xrest, ytrain, yrest = train_test_split(X, y, test_size=0.2)
Xval, Xtest, yval, ytest = train_test_split(Xrest, yrest, test_size=0.5)
lr.fit(Xtrain, ytrain)

columns = Xtrain.columns.values.tolist()
coefs = lr.coef_.ravel().tolist()

plt.figure(figsize=(10, 6))
plt.barh(columns, coefs)
plt.xlabel("Feature importance")
plt.ylabel("Features")

lr.fit(Xtrain, ytrain)

def calculate_relative_error(rmse, y):
    y_range = y.max() - y.min()
    relative_error = float((rmse / y_range) * 100)

    return relative_error

y_pred_lr = lr.predict(Xtest)
rmse_adr = RMSE(yval, y_pred_lr)
relative_error = calculate_relative_error(rmse_adr, y)

print("Relative error for the average daily revenue: {:.2f}%".format(relative_error))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = cross_val_score(lr, Xtrain, ytrain, cv=kf, scoring='neg_root_mean_squared_error')
print(" Cross validated RMSE:", -rmse_scores.mean())

RMSE(yval, y_pred_lr)

xgb = xgboost.XGBRegressor()
xgb.fit(Xtest, ytest)

y_pred_xgb = xgb.predict(Xtest)
rmse_adr = RMSE(yval, y_pred_xgb)
relative_error = calculate_relative_error(rmse_adr, y)

print("Relative error for the average daily revenue: {:.2f}%".format(relative_error))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = cross_val_score(xgb, Xtrain, ytrain, cv=kf, scoring='neg_root_mean_squared_error')

print("Cross validated RMSE:", -rmse_scores.mean())

RMSE(yval, y_pred_xgb)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(Xtrain, ytrain)

y_pred_knn = knn.predict(Xtest)
rmse_adr = RMSE(yval, y_pred_knn)
relative_error = calculate_relative_error(rmse_adr, y)

print("Relative error for the average daily revenue: {:.2f}%".format(relative_error))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = cross_val_score(knn, Xtrain, ytrain, cv=kf, scoring='neg_root_mean_squared_error')

print("Cross validated RMSE:", -rmse_scores.mean())

RMSE(yval, y_pred_knn)

def predict_knn(dataframe):
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.predict(dataframe)