# source myenv/bin/activate (activating Virtual Enviroment for for non deb packages)


# Importing Library
import pandas as pd
import seaborn as sns
import numpy as np 
from matplotlib import pyplot as plt

# Reading CSV and printing Output
df = pd.read_csv('Permdata.csv')
print(df)
print(df.columns)

# Calling inputs and target attributes
inputs = df.drop('LITHOLOGY',axis='columns')
target=df.LITHOLOGY

# Printing Input and Target
print(inputs)
print(target)

# Desc of Dataset
print(df.describe())
print(inputs.corr())

# Generating Coorelation heatmap
sns.heatmap(inputs.corr(),cmap='Spectral',annot=True,vmin=-1,vmax=1)
# Showing Heatmap
plt.show()
