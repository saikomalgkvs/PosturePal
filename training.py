import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the datasets
def train_model(bad_posture_data, good_posture_data):

    # Add labels to the datasets
    for bad_df in bad_posture_data:
        bad_df['label'] = 'bad'
    
    for good_df in good_posture_data:
        good_df['label'] = 'good'
        
    # Combine the datasets
    bad_posture_data = pd.concat(bad_posture_data, ignore_index=True)
    good_posture_data = pd.concat(good_posture_data, ignore_index=True)

    data = pd.concat([bad_posture_data, good_posture_data], ignore_index=True)

    # Separate features and labels
    X = data.drop(columns=['label'])
    y = data['label']

    # Encode labels
    y = y.map({'bad': 0, 'good': 1})

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    accuracy = accuracy_score(y_test, y_pred)

    # return the trained model and accuracy
    return model, accuracy