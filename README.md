import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

sales_data = pd.read_csv('sales_data.csv')
ingredients_data = pd.read_csv('ingredients_data.csv')

sales_data['Date'] = pd.to_datetime(sales_data['Date'])
sales_data.set_index('Date', inplace=True)
weekly_sales = sales_data.resample('W').sum()

train_data = weekly_sales[:-4]
test_data = weekly_sales[-4:]

model = SARIMAX(train_data['Quantity Sold'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
model_fit = model.fit()

predictions = model_fit.forecast(steps=4)
mape = mean_absolute_percentage_error(test_data['Quantity Sold'], predictions)

ingredient_requirements = {}
for index, row in ingredients_data.iterrows():
    ingredient_requirements[row['Pizza Type']] = row['Quantity Needed']

predicted_sales = predictions.values
purchase_order = {}

for i in range(len(predicted_sales)):
    pizza_type = sales_data['Pizza Type'].unique()[i % len(sales_data['Pizza Type'].unique())]
    required_quantity = predicted_sales[i] * ingredient_requirements[pizza_type]
    purchase_order[pizza_type] = required_quantity

purchase_order_df = pd.DataFrame(list(purchase_order.items()), columns=['Pizza Type', 'Quantity Needed'])
purchase_order_df.to_csv('purchase_order.csv', index=False)

print(f'MAPE: {mape}')
print(purchase_order_df)
