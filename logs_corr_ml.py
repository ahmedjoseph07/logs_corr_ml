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
# print(df)
# print(df.columns)



# Calling inputs and target attributes
encoder = LabelEncoder() # Label encoding the target data
df['LITHOLOGY_encoded'] = encoder.fit_transform(df['LITHOLOGY']) # Unique values are encoded with a number (fit and transform)
inputs = df.drop(['LITHOLOGY','LITHOLOGY_encoded'],axis='columns')
target=df['LITHOLOGY_encoded']
# print(df[['LITHOLOGY', 'LITHOLOGY_encoded']].head(7))
print(encoder.classes_) # Unique values of the target data (Return the classes labels as a list)
print(dict(zip(encoder.classes_, range(len(encoder.classes_))))) # Create a dictionary for printing what lithology is encoded as what number


# Printing Input and Target
# print(inputs)
# print(target)

# Descrption of Dataset
# print(df.describe())
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
from sklearn.model_selection import GridSearchCV

model=RandomForestRegressor()

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
