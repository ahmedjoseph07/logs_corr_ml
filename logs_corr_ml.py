# PEF --> Photoelectric Factor (PEF)
# GR --> Gamma Ray (GR)
# SP --> Spontaneous Potential (SP)
# CAL --> Caliper (CAL)

# source myenv/bin/activate (activating Virtual Enviroment for for non deb packages)


# Importing Library
import pandas as pd
import seaborn as sns
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Reading CSV and printing Output
df = pd.read_csv('Permdata.csv')
print(df)
print(df.columns)



# Calling inputs and target attributes
encoder = LabelEncoder() # Label encoding the target data
df['LITHOLOGY_encoded'] = encoder.fit_transform(df['LITHOLOGY']) # Unique values are encoded with a number (fit and transform)
inputs = df.drop(['LITHOLOGY','LITHOLOGY_encoded'],axis='columns')
target=df['LITHOLOGY_encoded']
print(df[['LITHOLOGY', 'LITHOLOGY_encoded']].head(7))
print(encoder.classes_) # Unique values of the target data (Return the classes labels as a list)
print(dict(zip(encoder.classes_, range(len(encoder.classes_))))) # Create a dictionary for printing what lithology is encoded as what number


# Printing Input and Target
print(inputs) # Features
print(target) # Label

# Descrption of Dataset
print(df.describe())
print(inputs.corr())

# Generating Coorelation heatmap
sns.heatmap(inputs.corr(),cmap='Spectral',annot=True,vmin=-1,vmax=1)
# Showing Heatmap
# plt.show() # Heatmap of correlation matrix will be shown in the output

# Importing train test module to separate training and testing data

from sklearn.model_selection import train_test_split # Need to isntall sklearn by this command(pip install scikit-learn)

# Alloting training and testing data, here 70% data has been alloted for (training/*testing) the model 
x_train, x_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3,random_state=1) #x is for features and y is for label(Lithology)

# Calling Random Forest Regressor model
from sklearn.ensemble import RandomForestRegressor 
# from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV # GridSearchCV is used to find the best hyperparameters (tuning) of the model

model=RandomForestRegressor() #Calling Regressor model

# model = RandomForestClassifier() #Calling Classifier model 

# Training the model with train data set 
model.fit(x_train,y_train)



# Calculating Correlation Coefficient of the model from test data set
corrCoff = model.score(x_test,y_test)
print(f"Correlation Coefficient:{corrCoff}") #Return the coefficient of determination R^2 of the prediction.

# Visualizing predicted variables
# y_pred = model.predict(inputs) (We should not use this as it will predict the same data that was used for training)
y_pred = model.predict(x_test) # Predicting the model with test data set (generated from train_test_split)
print(y_pred)

y_pred_rounded = np.round(y_pred) # Rounding off the predicted values to the nearest integer
print(y_pred_rounded) # Displaying the rounded predictions

lithology_predicted = encoder.classes_[y_pred_rounded.astype(int)] # Mapping the rounded predictions to their corresponding lithology labels
print(lithology_predicted) # Displaying the predicted lithology labels

# Calculating the accuracy of the model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
accuracy = accuracy_score(encoder.classes_[y_test], lithology_predicted)
print(f"Accuracy: {accuracy}")  #Classifier model is more suitable for this dataset

# Calculating mean squared error (must use regrrssor model as it works on continuous data)

from sklearn.metrics import mean_squared_error
# y_test = [3, 2, 1, 0, 2]    # Actual labels (encoded)
# y_pred = [2.8, 2.1, 0.9, 0.2, 2.4]  # Predicted values from the regressor
MSE = mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error: {MSE}")

# Calculating Root Mean Squared Error (must use regressor model as it  works on continuous data)
RMSE = np.sqrt(MSE)
print(f"Root Mean Squared Error: {RMSE}")

#Hyper-parameter of the RandomForest Model
print(model.get_params())

#Importing Decision Tree Regre
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
model_T = DecisionTreeRegressor()

#Training the model with train data set
model_T.fit(x_train,y_train)

# Predicted target variables
y_pred_T = model_T.predict(x_test) # Predicting the model with test data set (generated from train_test_split)

print(y_pred_T)

# Calculating Correlation Coefficient of the model from test data set
corrCoff_T = model_T.score(x_test,y_test)
print(f"Correlation Coefficient:{corrCoff_T}") #Return the coefficient of determination R^2 of the prediction.

# Calculating mean squared error (must use regrrssor model as it works on continuous data)
MSE_T = mean_squared_error(y_test,y_pred_T)
print(f"Mean Squared Error: {MSE_T}")

# Calculating Root Mean Squared Error (must use regressor model as it  works on continuous data)
RMSE_T = np.sqrt(MSE_T)
print(f"Root Mean Squared Error: {RMSE_T}")

# Finding the feature importance
feature_names = inputs.columns
print(feature_names)
print(model.feature_importances_)

# Generating tree diagram
fig = plt.figure(figsize=(50,40))
_ = tree.plot_tree(model_T, feature_names=feature_names, filled=True,fontsize=6)
plt.show()

# Hyper Parameter of DT model 
print(model_T.get_params())

# Importing Xtreme Gradient Boosting Model 
from sklearn.model_selection import GridSearchCV
import xgboost as xgb #(pip install xgboostr)

# model_xgb = xgb.XGBClassifier()
model_xgb = xgb.XGBRegressor()

# Training the model with train data set
model_xgb.fit(x_train,y_train)

# Calculating Correlation Coefficient of the model from test data set
print(f"Correlation Coefficient:{model_xgb.score(x_test,y_test)}")

# Visualizing of the predicted target values
y_pred_xgb = model_xgb.predict(x_test) # Predicting the model with test data set (generated from train_test_split)\
print(y_pred_xgb)

# Calculating mean squared error of XGB Model 
MSE_xgb = mean_squared_error(y_test,y_pred_xgb)
print(f"Mean Squared Error: {MSE_xgb}")

#Calculating Root Mean Squared Error of XGB Model
RMSE_xgb = np.sqrt(MSE_xgb)
print(f"Root Mean Squared Error: {RMSE_xgb}")

# Hyper Parameter of XGB Model
print(model_xgb.get_params())

