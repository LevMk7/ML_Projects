Sales Forecasting Project
This project focuses on forecasting daily sales of stickers across multiple countries using advanced machine learning models. I explored several models and preprocessing techniques to improve the accuracy of predictions.

Goal
The goal is to predict the number of items sold on each day across various countries and stores, with the best possible accuracy.

Approach
Data Preparation
The dataset was preprocessed by filling missing values, creating new features, and encoding categorical variables such as country, store, and product.
We also extracted date-related features such as year, month, month_sin, and month_cos for better model performance.

Modeling
I trained multiple models, including:

LightGBM
XGBoost
Neural Networks (Keras)

Best Result
The best result was achieved with the XGBoost, with an impressive ~3000 MSE on the validation data.

Visualizations
Here is an analysis of sales trends by country and plotted daily sales with 30-day moving averages for better insights.