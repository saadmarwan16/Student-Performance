# Import the necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import pandas as pd

# Load the data into a pandas dataframe and remove outliers
files = glob("dataset/*.csv")
df = pd.concat((pd.read_csv(file, sep=";") for file in files), ignore_index=True)
df = df[(abs(df["G1"] - df["G3"])) < 7]
df = df[(abs(df["G1"] - df["G2"])) < 7]
df = df[(abs(df["G2"] - df["G3"])) < 7]

# Put the data into training and testing sets
evidence = df[["G1", "G2"]]
label = df[["G3"]]
X_train, X_test, y_train, y_test = train_test_split(
                                                    np.array(evidence),
                                                    np.array(label),
                                                    test_size=0.4)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

# Print the results
print((r2_score(y_test, prediction)))