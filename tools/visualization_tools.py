import matplotlib.pyplot as plt
import torch


def plot_training_statistics(
        training_stats, 
        title="Training Statistics", 
        zoomed=True, 
        saving_path='./training-curves/training_statistics.png'):
    """
    Plots training statistics such as loss, accuracy, and F1-score over epochs.

    Parameters:
    - training_stats (dict): A dictionary containing lists of training statistics with keys:
        'train_losses', 'train_accuracies', 'train_f1s', 'val_losses', 'val_accuracies', 'val_f1s'.
    """

    labels_graphs_training = ['Train Loss', 'Train Accuracy', 'Train F1-score', 'Train ROC-AUC']
    labels_graphs_validation = ['Test Loss', 'Test Accuracy', 'Test F1-score', 'Test ROC-AUC']
    columns_names_training = ['train_losses', 'train_accuracies', 'train_f1s', 'train_roc_aucs']
    columns_names_validation = ['test_losses', 'test_accuracies', 'test_f1s', 'test_roc_aucs']

    # Tracer les courbes
    plt.figure(figsize=(12, 5))
    plt.title('Training and Validation Metrics Zoomed for fine tunned resnet50')
    plt.delaxes(plt.gca())  # Supprime l'axe vide créé par plt.title()

    for i in range(1, 5):
        plt.subplot(1, 4, i)
        plt.plot(training_stats[columns_names_training[i-1]], label=labels_graphs_training[i-1])
        plt.plot(training_stats[columns_names_validation[i-1]], label=labels_graphs_validation[i-1])
        plt.xlabel('Epoch')
        plt.ylabel(labels_graphs_training[i-1].split()[1])  # Utilise le nom de la métrique comme label
        if not zoomed:
            plt.ylim(0, 1)
        plt.legend()

    plt.tight_layout()
    plt.savefig(saving_path)
    plt.show()


def print_training_statistics(
        train_loss, train_accuracy, train_f1, train_roc_auc, 
        test_loss, test_accuracy, test_f1, test_roc_auc,
        get_device_properties=False, memory_allocated=False, memory_reserved=False): 
    """ Prints training statistics such as loss, accuracy, and F1-score for each epoch. 
    Parameters: 
    - training_stats (dict): A dictionary containing lists of training statistics with keys: 'train_losses', 'train_accuracies', 'train_f1s', 'val_losses', 'val_accuracies', 'val_f1s'. """
    print('-' * 96)
    print(f"| Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Train F1-score: {train_f1:.4f} | Train ROC-AUC: {train_roc_auc:.4f} |")
    print(f"| Test Loss:  {test_loss:.4f} | Test Accuracy:  {test_accuracy:.4f} | Test F1-score:  {test_f1:.4f} | Test ROC-AUC:  {test_roc_auc:.4f} |")
    if get_device_properties or memory_allocated or memory_reserved:
        gdp = torch.cuda.get_device_properties(0).total_memory / 1024**3
        ma = torch.cuda.memory_allocated() / 1024**3
        mr = torch.cuda.memory_reserved() / 1024**3
        print(f"| GPU VRAM checks -> | total: {gdp:.2f} Go        | used: {ma:.2f} Go          | cached: {mr:.2f} Go       |")
    print('-' * 96)


# import matplotlib.pyplot as plt

# def imshow(img):
#     img = img.numpy().transpose((1, 2, 0))
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     img = std * img + mean
#     img = np.clip(img, 0, 1)
#     plt.imshow(img)
#     plt.show()

# # Exemple d'affichage
# images, labels = next(iter(train_loader))
# imshow(images[0])
# print("Labels:", labels[0])