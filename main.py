import pandas as pd
from sklearn.model_selection import train_test_split
from decision_tree import Decision_Tree

# Load data
data = pd.read_csv("PCOS.csv")

X = data.drop("PCOS (Y/N)", axis=1)  # Features
y = data[["PCOS (Y/N)"]]  # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # Train-test split

# Train and evaluate model
model = Decision_Tree(max_depht=5)  # Initialize model
model.fit_model(X_train, y_train)  # Fit the model
y_predict = model.make_predict(X_test)  # Make predictions
accuracy = model.score(y_predict, y_test)  # Calculate accuracy
print(accuracy)  # Print accuracy
