import matplotlib.pyplot as plt
import pandas as pd
import os

# Change directory
os.chdir("C:/Users/levmk/Desktop/KaggleProject/preparations")  # Replace with your desired path

train = pd.read_csv("train.csv", encoding='utf8')
train = train.fillna(0)

def plot_country_sales(country_name):
    # Filter data for the specified country
    country_data = train[train['country'] == country_name].copy()

    if country_data.empty:
        print(f"No data available for country '{country_name}'.")
        return
    
    # Assuming we have a 'num_sold' column
    country_data['date'] = pd.to_datetime(country_data['date'])
    country_daily = country_data.groupby('date')['num_sold'].sum()

    # Calculate 30-day moving average
    rolling_mean = country_daily.rolling(window=30).mean()

    # Plot the data
    plt.figure(figsize=(15, 6))
    plt.plot(country_daily.index, country_daily.values, label='Daily Sales', color='blue', alpha=0.7)
    plt.plot(rolling_mean.index, rolling_mean.values, label='30-Day Moving Average', color='red', linewidth=2)
    plt.fill_between(country_daily.index, country_daily.values, rolling_mean.values, color='blue', alpha=0.1)

    # Customize the plot
    plt.title(f'Sales Trend for {country_name} with 30-Day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

plot_country_sales("Canada")
plot_country_sales("Norway")
plot_country_sales("Italy")