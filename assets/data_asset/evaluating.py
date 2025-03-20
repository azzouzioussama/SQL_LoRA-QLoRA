import torch
from datasets import load_dataset, load_metric
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()



class MetricsCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        y_pred = [output['predicted_label'] for output in kwargs['eval_dataloader']]
        y_true = kwargs['eval_dataloader'].dataset['labels']
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        logloss = log_loss(y_true, y_pred)
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'Log Loss: {logloss}')


from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm



# Function to calculate accuracy
def calculate_accuracy(model, dataloader, device):
    """
    Calculates the accuracy of the model on the given dataloader.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the evaluation dataset.
        device (torch.device): The device on which the model and data are loaded.

    Returns:
        float: The accuracy of the model on the dataset.
    """
    model.eval()  # Set the model to evaluation mode
    metric = evaluate.load("accuracy")  # Load accuracy metric
    
    # Iterate through the dataloader
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to device
        
        with torch.no_grad():
            output_tokens = model(**batch)  # Get model outputs
        
        logits = output_tokens.logits
        predictions = torch.argmax(logits, dim=-1)  # Get predictions
        
        # Flatten predictions and labels to 1D tensors for metric computation
        flattened_predictions = predictions.view(-1)
        flattened_labels = batch['labels'].view(-1)
        
        # Add batch to metric
        metric.add_batch(predictions=flattened_predictions, references=flattened_labels)
    
    # Compute and return accuracy
    accuracy = metric.compute()
    return accuracy["accuracy"]





