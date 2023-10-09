import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# getting data from csv
df1 = pd.read_csv('ad.csv')
df2 = pd.read_csv('ad (1).csv')
df3 = pd.read_csv('ad_10000records.csv')
frames = [df1, df2, df3]
data = pd.concat(frames)

#numerical values
data = data.replace('Male', 1)
data = data.replace('Female', 0)
data=data.rename(columns = {'Gender':'Male'})

# cleaning data set
x=data.iloc[:,0:7]
x=x.drop(['Ad Topic Line','City'],axis=1)
y=data.iloc[:,9]

# splitting dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#training and predicting
Lr=LogisticRegression(C=0.01,random_state=42)
Lr.fit(x_train,y_train)
y_pred=Lr.predict(x_test)

#printing accuracy and f1 score
print("Accuracy: %.2f" % (accuracy_score(y_test, y_pred)*100) + "%")
print("F1 score: %.4f" % f1_score(y_test,y_pred))