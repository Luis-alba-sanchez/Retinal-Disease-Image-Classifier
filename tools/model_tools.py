import torch
from tqdm import tqdm
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score, MultilabelAUROC
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC
from torch.amp import autocast


def train_one_epoch(model, train_loader, optimizer, criterion, device, num_labels, scaler=None):
    """
    Performs one epoch of training for the given model.
    
    :param model: Torch model to be trained (torch.nn.Module object)
    :param train_loader: DataLoader for training data (torch.utils.data.DataLoader object)
    :param optimizer: Optimizer for training (torch.optim object)
    :param criterion: Loss function for training (torch.nn object)
    :param device: Device to run training on (e.g., 'cuda' or 'cpu')
    :param scaler: GradScaler for Automatic Mixed Precision (optional)
    """
    model.train()
    running_loss = 0.0

    if num_labels == 1:
        accuracy_metric = BinaryAccuracy().to(device)
        f1_metric = BinaryF1Score().to(device)
        auroc_metric = BinaryAUROC().to(device)
    else:
        accuracy_metric = MultilabelAccuracy(num_labels=num_labels, average='micro').to(device)
        f1_metric = MultilabelF1Score(num_labels=num_labels, average='micro').to(device)
        auroc_metric = MultilabelAUROC(num_labels=num_labels, average='micro').to(device)

    for _, all in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = all[0].to(device), all[1].to(device)
        optimizer.zero_grad()
    
        # Use autocast for mixed precision if scaler is provided
        if scaler is not None:
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        # Use probabilities for metrics (not hard predictions)
        preds_probs = torch.sigmoid(outputs)
        labels_long = labels.long()  # Convert to int for torchmetrics
        accuracy_metric.update(preds_probs, labels_long)
        f1_metric.update(preds_probs, labels_long)
        auroc_metric.update(preds_probs, labels_long)

    train_loss = running_loss / len(train_loader)
    train_accuracy = accuracy_metric.compute().item()
    train_f1 = f1_metric.compute().item()
    train_roc_auc = auroc_metric.compute().item()

    # Reset metrics for next epoch
    accuracy_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    return train_loss, train_accuracy, train_f1, train_roc_auc


def validate(model, val_loader, criterion, device, num_labels):
    """
    Performs validation for the given model.
    
    :param model: Torch model to be validated (torch.nn.Module object)
    :param val_loader: DataLoader for validation data (torch.utils.data.DataLoader object)
    :param criterion: Loss function for validation (torch.nn object)
    :param device: Device to run validation on (e.g., 'cuda' or 'cpu')
    """
    model.eval()
    val_running_loss = 0.0

    if num_labels == 1:
        accuracy_metric = BinaryAccuracy().to(device)
        f1_metric = BinaryF1Score().to(device)
        auroc_metric = BinaryAUROC().to(device)
    else:
        accuracy_metric = MultilabelAccuracy(num_labels=num_labels, average='micro').to(device)
        f1_metric = MultilabelF1Score(num_labels=num_labels, average='micro').to(device)
        auroc_metric = MultilabelAUROC(num_labels=num_labels, average='micro').to(device)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            # Use probabilities for metrics (not hard predictions)
            preds_probs = torch.sigmoid(outputs)
            labels_long = labels.long()  # Convert to int for torchmetrics
            accuracy_metric.update(preds_probs, labels_long)
            f1_metric.update(preds_probs, labels_long)
            auroc_metric.update(preds_probs, labels_long)

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = accuracy_metric.compute().item()
    val_f1 = f1_metric.compute().item()
    val_roc_auc = auroc_metric.compute().item()

    return val_loss, val_accuracy, val_f1, val_roc_auc


def train_and_test_model(model, train_loader, test_loader, optimizer, criterion, device, num_labels, scaler=None):
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
    train_loss, train_accuracy, train_f1, train_roc_auc = train_one_epoch(model=model, train_loader=train_loader, optimizer=optimizer, criterion=criterion, device=device, num_labels=num_labels, scaler=scaler)
    
    # Testing on test set
    test_loss, test_accuracy, test_f1, test_roc_auc = validate(model=model, val_loader=test_loader, criterion=criterion, device=device, num_labels=num_labels)
    
    return train_loss, train_accuracy, train_f1, train_roc_auc, test_loss, test_accuracy, test_f1, test_roc_auc
