import torch
import pandas as pd
import numpy as np
import joblib  # to load the scaler and encoder
import torch.nn as nn
import scipy.sparse

# Define your model class (it should be the same as the one used for training)
class NFLPredictor(nn.Module):
    def __init__(self, input_size):
        super(NFLPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # Increased layer size
        self.fc2 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)  # Added dropout
        self.fc3 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(0.5)  # Added dropout
        self.fc4 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        return self.sigmoid(self.fc4(x))

# Load the trained model checkpoint
def load_model():
    model_checkpoint_path = 'model/nfl_predictor_model.pth'
    model = NFLPredictor(input_size=215)  # Adjust the number of input features to match your model
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()
scaler = joblib.load('model/scaler.joblib')  # Replace with your actual scaler file path
column_transformer = joblib.load('model/column_transformer.joblib')  # Replace with your actual column transformer file path

# Function to preprocess new data
def preprocess_data(data_frame):
    # Transform the data using the already-fitted column transformer
    transformed_data = column_transformer.transform(data_frame)
    
    # Check if the transformed data is sparse and convert to dense if necessary
    if scipy.sparse.issparse(transformed_data):
        transformed_data = transformed_data.toarray()
    
    # Scale the transformed data using the already-fitted scaler
    scaled_data = scaler.transform(transformed_data)

    # Convert to a PyTorch tensor
    tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
    return tensor_data

# Function to make predictions
def make_prediction(input_data):
    # Preprocess the input DataFrame
    preprocessed_data = preprocess_data(input_data)

    # Make predictions using the pre-trained model
    with torch.no_grad():
        predictions = model(preprocessed_data)

    # Return the prediction
    return predictions.cpu().numpy()