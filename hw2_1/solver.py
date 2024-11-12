import torch
import torch.nn as nn
import numpy as np

def train(model, criterion, optimizer, train_dataloader, num_epoch, device):
    """
    Main training loop for multiple epochs
    """
    model.to(device)
    avg_train_loss, avg_train_acc = [], []

    for epoch in range(num_epoch):
        model.train()
        batch_train_loss, batch_train_acc = train_one_epoch(model, criterion, optimizer, train_dataloader, device)
        avg_train_acc.append(np.mean(batch_train_acc))  # Averaging the accuracies
        avg_train_loss.append(np.mean(batch_train_loss))  # Averaging the losses

        print(f'\nEpoch [{epoch + 1}] Average training loss: {avg_train_loss[-1]:.4f}, '
              f'Average training accuracy: {avg_train_acc[-1]:.4f}')

    return model



def train_one_epoch(model, criterion, optimizer, train_dataloader, device):
    model.train()  # Set model to training mode
    total_loss = 0
    total_correct = 0

    # Initialize lists to store batch loss and accuracy
    batch_train_loss = []
    batch_train_acc = []

    # Loop over each batch in the DataLoader
    for inputs, targets in train_dataloader:
        # Move inputs and targets to the specified device (e.g., GPU if available)
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass: compute model predictions
        outputs = model(inputs)

        # Calculate the loss between outputs and one-hot encoded targets
        loss = criterion(outputs, targets)

        # Backward pass and optimizer step to update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy calculation
        # Convert one-hot encoded targets to class indices
        _, targets_class = torch.max(targets, 1)  # Convert one-hot targets to class indices
        _, predicted = torch.max(outputs, 1)      # Get predicted class index from model output

        # Sum up correct predictions for the batch
        correct = (predicted == targets_class).sum().item()  # Count correct predictions
        total_correct += correct
        total_loss += loss.item()

        # Calculate accuracy for the current batch
        batch_accuracy = correct / inputs.size(0)  # Accuracy for the current batch

        # Append batch loss and accuracy to lists
        batch_train_loss.append(loss.item())  # Store the current batch loss
        batch_train_acc.append(batch_accuracy)  # Store the current batch accuracy

    # Calculate average loss and accuracy over the entire epoch
    avg_loss = total_loss / len(train_dataloader)  # Average loss per batch
    accuracy = total_correct / len(train_dataloader.dataset)  # Accuracy over all examples

    # Append epoch-level average loss and accuracy to batch lists
    batch_train_loss.append(avg_loss)
    batch_train_acc.append(accuracy)

    # Return batch-wise loss and accuracy lists (including epoch averages)
    return batch_train_loss, batch_train_acc


def test(model, test_dataloader, device):
    """
    Testing loop to calculate accuracy on test data.
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation during inference
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate accuracy
            _, targets_class = torch.max(targets, 1)  # Convert one-hot targets to class indices
            _, predicted = torch.max(outputs, 1)      # Get predicted class index from model output

        # Sum up correct predictions for the batch
            correct = (predicted == targets_class).sum().item()  # Count correct predictions
            total_correct += correct

    # Calculate overall test accuracy
    avg_test_acc = total_correct / len(test_dataloader.dataset)
    print(f"The test accuracy is {avg_test_acc:.4f}.")

