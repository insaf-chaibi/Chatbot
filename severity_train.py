import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the dog diseases dataset
df1 = pd.read_csv("dataset2.csv")

# Define the input features (symptoms) and target variable (severity)
df = df1.iloc[:,1:-1]
symptoms = df.columns
severity = df1['Disease severity']

# Split the dataset into training and testing sets
X_train, X_test, y_sev_train, y_sev_test = train_test_split(df[symptoms], severity, test_size=0.3, random_state=42)

# Train decision tree models for severity prediction
sev_tree = DecisionTreeClassifier()
sev_tree.fit(X_train.values, y_sev_train)

# Make predictions on the test set
y_sev_pred = sev_tree.predict(X_test.values)

# Evaluate the accuracy of the models
sev_accuracy = accuracy_score(y_sev_test, y_sev_pred)

print("Severity prediction accuracy:", sev_accuracy)

# Save the trained model to disk
joblib.dump(sev_tree, "model/severity_model.joblib")

# Define a function to prompt the user for the severity of their symptoms
'''
def prompt_user():
    symptoms = [0 for i in range(len(df.columns[1:-1]))]
    
    for index, symptom in enumerate(df.columns[1:-1]):
        #symptoms.append(random.choice([0, 1]))

        response = input(f"Does your dog have {symptom.lower()}? (y/n): ")
        symptoms[index] = (1 if (response.lower() == 'y') else 0)

    predict(symptoms)
'''
    
# Define a function to make predictions using the decision tree models
def predict(symptoms):
    # Load the trained models from disk
    sev_tree = joblib.load("model/severity_model.joblib")
    
    # Make a prediction for the severity of the symptoms
    sev_pred = sev_tree.predict([symptoms])[0]
    sev_msg = f"The disease of your dog's symptoms is more likely {sev_pred}."

    # return sev_msg
    return sev_msg

    