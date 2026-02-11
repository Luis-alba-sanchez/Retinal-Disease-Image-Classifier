from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm



def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    Performs one epoch of training for the given model.
    
    :param model: Torch model to be trained (torch.nn.Module object)
    :param train_loader: DataLoader for training data (torch.utils.data.DataLoader object)
    :param optimizer: Optimizer for training (torch.optim object)
    :param criterion: Loss function for training (torch.nn object)
    :param device: Device to run training on (e.g., 'cuda' or 'cpu')
    """
    model.train()
    running_loss = 0.0
    train_preds = []
    train_targets = []

    for _, all in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = all[0].to(device), all[1].to(device)
        optimizer.zero_grad()
        outputs = model(images) # gets the logits (raw outputs) from the model
        loss = criterion(outputs, labels) # computes the loss between the model's (sigmoided) outputs and the true labels
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5 # applies sigmoid to get probabilities and then thresholds at 0.5 to get binary predictions
        train_preds.extend(preds.cpu().detach().numpy())
        train_targets.extend(labels.cpu().detach().numpy())

    train_loss = running_loss / len(train_loader)
    train_accuracy = accuracy_score(train_targets, train_preds)
    train_f1 = f1_score(train_targets, train_preds, average='micro')
    train_roc_auc = roc_auc_score(train_targets, train_preds, average='micro')

    return train_loss, train_accuracy, train_f1, train_roc_auc


def validate(model, val_loader, criterion, device):
    """
    Performs validation for the given model.
    
    :param model: Torch model to be validated (torch.nn.Module object)
    :param val_loader: DataLoader for validation data (torch.utils.data.DataLoader object)
    :param criterion: Loss function for validation (torch.nn object)
    :param device: Device to run validation on (e.g., 'cuda' or 'cpu')
    """
    model.eval()
    val_running_loss = 0.0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            val_preds.extend(preds.cpu().detach().numpy())
            val_targets.extend(labels.cpu().detach().numpy())

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average='micro')
    val_roc_auc = roc_auc_score(val_targets, val_preds, average='micro')

    return val_loss, val_accuracy, val_f1, val_roc_auc


def train_and_test_model(model, train_loader, test_loader, optimizer, criterion, device, batch_size=32, test_size=0.3):
    """
    Performs training and evaluation of the given model on the provided dataset.
    
    :param model: Torch model to be trained (torch.nn.Module object)
    :param dataset: Dataset to be used for training and evaluation (torch.utils.data.Dataset object)
    :param optimizer: Optimizer for training (torch.optim object)
    :param criterion: Loss function for training (torch.nn object)
    :param device: Device to run training and evaluation on (e.g., 'cuda' or 'cpu')
    :param batch_size: Batch size for training and evaluation (int)
    """
    
    # Train the model
    train_loss, train_accuracy, train_f1, train_roc_auc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    
    # Testing on test set
    test_loss, test_accuracy, test_f1, test_roc_auc = validate(model, test_loader, criterion, device)
    
    return train_loss, train_accuracy, train_f1, train_roc_auc, test_loss, test_accuracy, test_f1, test_roc_auc
