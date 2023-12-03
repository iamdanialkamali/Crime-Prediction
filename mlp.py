import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
from torch.utils.data import DataLoader, TensorDataset

# Check if CUDA is available
if torch.cuda.is_available():
    # Set the CUDA device
    device_id = 0  # Change this to the ID of the desired GPU
    torch.cuda.set_device(device_id)

    device = torch.device(f'cuda:{device_id}')
    print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")



# Logistic Regression Model Definition
class NeuralNetwork(nn.Module):
    def __init__(self, num_features,num_labels):
        super(NeuralNetwork, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),  # Batch Normalization after the first linear layer
            nn.ReLU(),            # ReLU activation
            nn.Dropout(0.2),      # Dropout for regularization

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),  # Batch Normalization after the second linear layer
            nn.ReLU(),            # ReLU activation
            nn.Dropout(0.2),      # Dropout for regularization

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),  # Batch Normalization after the third linear layer
            nn.ReLU(),            # ReLU activation
            nn.Dropout(0.2),      # Dropout for regularization

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),  # Batch Normalization after the fourth linear layer
            nn.ReLU(),            # ReLU activation
            nn.Dropout(0.2),      # Dropout for regularization

            nn.Linear(1024, num_labels)
            # No Batch Normalization or Dropout after the final layer
        )


    
    def forward(self, x):
        return self.linear(x)

def train_model(model, train_loader, learning_rate=0.01, num_epochs=100, penalty=None, C=1.0, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        tqdm_v = tqdm(train_loader)
        epoch_loss = 0
        for batch_index, (inputs, labels) in enumerate(tqdm_v):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if penalty == 'l1':
                l1_regularization = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l1_regularization += torch.norm(param, 1)
                loss += 1/C * l1_regularization
            elif penalty == 'l2':
                l2_regularization = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l2_regularization += torch.norm(param, 2)
                loss += 1/C * l2_regularization
            epoch_loss += loss.item()
            tqdm_v.set_description(str(loss))
            loss.backward()
            optimizer.step()
        
        print(f"epochs: {epoch + 1}, loss: {epoch_loss/ ( len(train_loader)/batch_size)} ")
        

# Prediction Function
def predict(model, X):
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(inputs)
        return outputs.round().numpy()

# Accuracy Calculation
def accuracy(model, X, y):
    y_pred = predict(model, X)
    correct = (y_pred.flatten() == y).sum()
    return correct / len(y)

# Sklearn-like Interface for Logistic Regression
class PyTorchMLP:
    def __init__(self, num_features, num_labels, learning_rate=0.01, max_iter=1_000_000, penalty=None, C=1.0, n_jobs=-1):
        self.model = NeuralNetwork(num_features, num_labels).to(device)
        self.learning_rate = learning_rate
        self.num_epochs = max_iter
        self.penalty = penalty
        self.C = C
    
    def fit(self, X, y, sample_weight=None):
        batch_size = 64  # Define your batch size

        # Convert training data to tensors
        X_train_tensor = torch.tensor(X, dtype=torch.float32)
        y_train_tensor = torch.tensor(y, dtype=torch.long)

        # Create a Dataset and DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Now pass train_loader to your training function

        train_model(self.model, train_loader, self.learning_rate, self.num_epochs, self.penalty, self.C, batch_size)

    def predict(self, X):
        return predict(self.model, X)

    def score(self, X, y):
        return accuracy(self.model, X, y)

