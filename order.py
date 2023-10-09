import numpy
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as ltb

# dowload data
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/supplement.csv")

# plotting pie chart
pie = data["Store_Type"].value_counts()
store = pie.index
orders = pie.values
fig = px.pie(data, values=orders, names=store)
# fig.show()

# plotting pie chart
pie2 = data["Location_Type"].value_counts()
location = pie2.index
orders = pie2.values
fig = px.pie(data, values=orders, names=location)
# fig.show()

# plotting pie chart
pie3 = data["Discount"].value_counts()
discount = pie3.index
orders = pie3.values
fig = px.pie(data, values=orders, names=discount)
# fig.show()

# plotting pie chart
pie4 = data["Holiday"].value_counts()
holiday = pie4.index
orders = pie4.values
fig = px.pie(data, values=orders, names=holiday)
# fig.show()

# transform to numerical values
data["Discount"] = data["Discount"].map({"No": 0, "Yes": 1})
data["Store_Type"] = data["Store_Type"].map({"S1": 1, "S2": 2, "S3": 3, "S4": 4})
data["Location_Type"] = data["Location_Type"].map({"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5})
data.dropna()

# creating arrays
x = np.array(data[["Store_Type", "Location_Type", "Holiday", "Discount"]])
y = np.array(data["#Order"])


# splitting dataset
xtrain, xtest, ytrain, ytest = train_test_split(x,
                                                y, test_size=0.3,
                                                random_state=42)

# training of model
model = ltb.LGBMRegressor()
model.fit(xtrain, ytrain)

# predicting and florring values of order
ypred = model.predict(xtest)
ypred = numpy.floor(ypred)

# calculating of avarage error
result = numpy.subtract(ypred, ytest)
result = abs(result)
sumOfErrors = numpy.sum(result)
meanError = (sumOfErrors/result.size)
print("Průměrná odchylka objednávek: " + str(meanError))

#calculating root of mean squared error
mse = mean_squared_error(ytest, ypred)
rmse = mse**(0.5)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % rmse)

# plotting predicted and actual number of orders
ytest = ytest[:100]
ypred = ypred[:100]
x_ax = range(len(ytest))
plt.figure(figsize=(12, 6))
plt.plot(x_ax, ytest, label="original")
plt.plot(x_ax, ypred, label="predicted")
plt.title("Number of order dataset test and predicted data")
plt.xlabel('X')
plt.ylabel('Price')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(True)
plt.show()
