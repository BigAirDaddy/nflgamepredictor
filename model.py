import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import scipy.sparse
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
df = pd.read_csv('data/nfl_games.csv')

# Convert 'date' to datetime and extract relevant features (e.g., year)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# Select relevant features and target
features = df[['season', 'neutral', 'playoff', 'team1', 'team2', 'elo1', 'elo2', 'elo_prob1']]
target = df['outcome']

# One-hot encode categorical variables
categorical_features = ['team1', 'team2']
numerical_features = ['season', 'neutral', 'playoff', 'elo1', 'elo2', 'elo_prob1']

# Create a column transformer with one-hot encoding for categorical features
transformer = ColumnTransformer(
    transformers=[
        ('one_hot', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

# Fit the transformer on the features
transformed_features = transformer.fit_transform(features)

# Save the fitted transformer for later use
joblib.dump(transformer, 'model/column_transformer.joblib')

# Scale the features
scaler = StandardScaler(with_mean=False)
scaled_features = scaler.fit_transform(transformed_features)
# Convert the scaled features to a dense array if they are in sparse format

if scipy.sparse.issparse(scaled_features):
    scaled_features = scaled_features.toarray()

# Save the fitted scaler for later use
joblib.dump(scaler, 'model/scaler.joblib')


# Convert to PyTorch tensors
X = torch.tensor(scaled_features, dtype=torch.float).to(device)
y = torch.tensor(target.values, dtype=torch.float).unsqueeze(1).to(device)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch Dataset
class NFLGamesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create data loaders
train_dataset = NFLGamesDataset(X_train, y_train)
test_dataset = NFLGamesDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=30, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=30, shuffle=False)

# Neural network architecture
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
# Determine the input size from the transformed features
input_size = scaled_features.shape[1]
model = NFLPredictor(input_size).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter('runs/nfl_predictor_experiment_1')

model.train()
num_epochs = 10  # Adjust this to your desired number of epochs
for epoch in range(num_epochs):
    total_loss = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Log the loss to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

    # Print average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss}')

# Close the TensorBoard writer
writer.close()
# Close the TensorBoard writer
writer.close()

# Close the TensorBoard writer
writer.close()
# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = (outputs.data > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy}')

# Save the model
torch.save(model.state_dict(), 'model/nfl_predictor_model.pth')
