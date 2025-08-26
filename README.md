# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Sanjay Sivaramakrishnan M 

### Register Number: 212223240151

```python
class Model(nn.Module):
      def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features,out_features)

    def forward(self, x):
        return self.linear(x)
# Initialize the Model, Loss Function, and Optimizer

torch.manual_seed(59)  # Ensure same initial weights
model = Model(1, 1)
initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()
print("\nName: Sanjay Sivaramakrishnan M")
print("Register No: 212223240151")
print(f'Initial Weight: {initial_weight:.8f}, Initial Bias: {initial_bias:.8f}\n')
# Define Loss Function & Optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
# Train the Model
epochs = 50
losses = []
for epoch in range(1, epochs + 1):  # Loop over epochs
    y_pred = model(X)
    loss = loss_function(y_pred,y)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.linear.weight.item():10.8f}  '
          f'bias: {model.linear.bias.item():10.8f}')


```

### Dataset Information
<img width="1393" height="282" alt="image" src="https://github.com/user-attachments/assets/cd57e469-d8dc-49b0-8f80-524e316003e4" />
<img width="1051" height="263" alt="image" src="https://github.com/user-attachments/assets/dfd4c8f8-202b-4bc1-9469-654a35983245" />
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/1d0144e8-9210-4426-a47c-a12f0dd7b2ad" />


### OUTPUT
#### Training Loss Vs Iteration Plot
<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/8331509c-7d04-4401-a28c-a6cc8d40407d" />

#### Best Fit line plot
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/b06ffc9d-fc3a-447a-9db5-d4226b95b01d" />


### New Sample Data Prediction
<img width="1544" height="418" alt="image" src="https://github.com/user-attachments/assets/80e69e42-2cb2-4fb5-be88-0dd7b2fa6493" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
