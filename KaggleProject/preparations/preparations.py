import pandas as pd
import numpy as np
import holidays
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
# Change directory
os.chdir("C:/Users/levmk/Desktop/KaggleProject/preparations")  # Replace with your desired path
print("New Directory:", os.getcwd())

train = pd.read_csv("../preparations/train.csv", encoding='utf8')
test = pd.read_csv("../preparations/test.csv", encoding='utf8')
train = train.fillna(0)
test = test.fillna(0)

train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

alpha2 = dict(zip(np.sort(train.country.unique()), ['CA', 'FI', 'IT', 'KE', 'NO', 'SG']))
h = {c: holidays.country_holidays(a, years=range(2010, 2020)) for c, a in alpha2.items()}

train['is_holiday'] = 0
test['is_holiday'] = 0

for c in alpha2:
    # Extract holiday dates from h[c] as a list of datetime objects
    holiday_dates = pd.to_datetime(list(h[c].keys()))  
    train.loc[train.country == c, 'is_holiday'] = train['date'].isin(holiday_dates).astype(int)
    test.loc[test.country == c, 'is_holiday'] = test['date'].isin(holiday_dates).astype(int)

# Encoding categorical features
categorical_features = ['country', 'store', 'product']

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()                         # Object that converts categorical values (e.g., text labels) into numerical values
    train[col] = le.fit_transform(train[col])   # Finds all unique values in the column train[col] and converts them into numerical format
    test[col] = le.transform(test[col])         # Repeats for test
    label_encoders[col] = le                    # Stores the LabelEncoder object for reuse

# Additional processing since the date was in the format 2010-01-01
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['day_of_week'] = train['date'].dt.dayofweek
train['day_of_year'] = train['date'].dt.dayofyear

test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['day_of_week'] = test['date'].dt.dayofweek
test['day_of_year'] = test['date'].dt.dayofyear

# Adding trigonometric functions since the days of the year repeat cyclically
# For example, January 1st of any year is New Year's Day
train['day_sin'] = np.sin(2 * np.pi * train['day_of_year'] / 365.0)
train['day_cos'] = np.cos(2 * np.pi * train['day_of_year'] / 365.0)
train['month_sin'] = np.sin(2 * np.pi * train['month'] / 12.0)
train['month_cos'] = np.cos(2 * np.pi * train['month'] / 12.0)
train['day_of_week_sin'] = np.sin(2 * np.pi * train['day_of_week'] / 7.0)
train['day_of_week_cos'] = np.cos(2 * np.pi * train['day_of_week'] / 7.0)

test['day_sin'] = np.sin(2 * np.pi * test['day_of_year'] / 365.0)
test['day_cos'] = np.cos(2 * np.pi * test['day_of_year'] / 365.0)
test['month_sin'] = np.sin(2 * np.pi * test['month'] / 12.0)
test['month_cos'] = np.cos(2 * np.pi * test['month'] / 12.0)
test['day_of_week_sin'] = np.sin(2 * np.pi * test['day_of_week'] / 7.0)
test['day_of_week_cos'] = np.cos(2 * np.pi * test['day_of_week'] / 7.0)

# Selecting features for training
features = ['country', 'store', 'product', 'year', 'month', 'day', 
            'day_sin', 'day_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
X = train[features]
y = train['num_sold']

# Splitting into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("preparations completed!")