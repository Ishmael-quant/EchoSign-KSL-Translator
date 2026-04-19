import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# LOAD DATA 
data = pd.read_csv("dataset.csv", header=None)

 #FILTER LABELS 
allowed_labels = ['A', 'B', 'C', 'L', 'Y', 'F', 'T']
data = data[data.iloc[:, -1].isin(allowed_labels)]
data = pd.read_csv("dataset.csv", header=None)

 #SPLIT FEATURES & LABELS 
X = data.iloc[:, :-1]   # all columns except last
y = data.iloc[:, -1]    # last column (labels)

#TRAIN TEST SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#TRAIN MODEL 
model = RandomForestClassifier()
model.fit(X_train, y_train)

 #EVALUATE 
accuracy = model.score(X_test, y_test)
print("✅ Accuracy:", accuracy)

#SAVE MODEL 
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")