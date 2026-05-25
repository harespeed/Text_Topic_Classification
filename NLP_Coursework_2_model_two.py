#!/usr/bin/env python
# coding: utf-8

# # Machine Learning for NLP COURSEWORK 2 model two
# ## Make sure to add all training, validation and testing dataset to the root and make sure to setup GPU otherwise its going to take a lot of time to train.
# 
# ## Group members:
# ### - Naeem Ali
# ### - Qiang Fu
# ### - Qiran Sun
# ### - Yanzong Qu
# ### - Zhaoliang Liu

# # Libraries

# In[1]:


get_ipython().system('pip install datasets transformers')

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay


# # Loading our datasets

# In[ ]:


"""
Reading training, validation and test datasets
"""
train_df = pd.read_csv('topic-train.csv')
valid_df = pd.read_csv('topic-validation.csv')
test_df = pd.read_csv('topic-test.csv')

"""
Converting string labels to list of floats
"""
train_df["labels"] = train_df['gold_label_list'].apply(lambda x: list(map(float, x.strip("[]").split())))
valid_df["labels"] = valid_df['gold_label_list'].apply(lambda x: list(map(float, x.strip("[]").split())))
test_df["labels"] = test_df['gold_label_list'].apply(lambda x: list(map(float, x.strip("[]").split())))

"""
Converting our dataset to dataset table to access methods like map
"""
train_data = Dataset.from_pandas(train_df)
val_data = Dataset.from_pandas(valid_df)
test_data = Dataset.from_pandas(test_df)


# # Preprocessing the dataset

# In[ ]:


"""
We are using bert-base-uncased model as our Language Model which is pretrained
for classification tasks and we are going to fine-tune it for topic classification.
"""
model_name = "bert-base-uncased"

"""
The AutoTokenizer will select the best tokenizer for our model.
"""
tokenizer = AutoTokenizer.from_pretrained(model_name)

"""
This method will tokenize our dataset.
"""
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True)

"""
Tokenizing our datasets
"""
tokenized_dataset_train = train_data.map(tokenize_function, batched=True)
tokenized_dataset_valid = val_data.map(tokenize_function, batched=True)
tokenized_dataset_test = test_data.map(tokenize_function, batched=True)


# # Initializing the language model

# In[ ]:


"""
Checking if GPU is available, then use it for faster training, otherwise it will use CPU.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Defining all labels in our datasets, used to get total number of labels and for error analysis
"""
all_labels = ["arts & culture", "business & entrepreneurs", "celebrity & pop culture",
              "diaries & daily life", "family", "fashion & style", "film tv & video",
              "fitness & health", "food & dining", "gaming",
              "learning & educational", "music", "news & social concern", "other hobbies",
              "relationships", "science & technology", "sports", "travel & adventure",
              "youth & student life"]

NUM_LABELS = len(all_labels)

"""
Initializing our model using AutoModelForSequenceClassification
"""
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS).to(device)


# # Setting up training arguments and training the model

# In[ ]:


"""
Setting up the training arguments.
- output_dir: is going to output the checkpoints in this folder location provided
- num_train_epochs: numbers of epochs to train the model
- per_device_train_batch_size: number of tweets per batch during training
- per_device_eval_batch_size: number of tweets per batch during training
- weight_decay: regularization parameter
- logging_dir: path to store model logs
- eval_strategy: control how often model should be evaluate
- learning_rate: step size for optimization process
- report_to: to report the model training process to external resource
- seed: it constrol the random state to produce the results in the same way
"""
training_args = TrainingArguments(
    output_dir=f"{model_name}-fine-tuned",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir=f"{model_name}-logs",
    eval_strategy="epoch",
    learning_rate=2e-5,
    report_to="none",
    seed=42,
)

"""
method to compute the evaluation metrics.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits > 0).astype(int)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

"""
Setting up the trainer.
- model: the model to train
- args: training arguments
- train_dataset: training dataset
- eval_dataset: validation dataset
- processing_class: tokenizer
- compute_metrics: evaluation function
"""
trainer = Trainer(
    model=model,                            # the model to train
    args=training_args,                     # training arguments
    train_dataset=tokenized_dataset_train,  # training dataset
    eval_dataset=tokenized_dataset_valid,   # validation dataset (use training for demo)
    processing_class=tokenizer,             # tokenizer
    compute_metrics=compute_metrics         # evaluation function
)

"""
Finally, training the model.
"""
trainer.train()


# # Evaluation metrics for the trained model

# In[ ]:


eval_results = trainer.evaluate()
print(eval_results)


# # Testing our fine-tuned model on the testing dataset

# In[ ]:


"""
Using our test dataset to evaluate the model
"""
y_pred_test = trainer.predict(tokenized_dataset_test)


# # Error analysis

# In[ ]:


"""
Because our model return the probabilities, we need to convert them to binary
predictions for error analysis
"""

# Convert logits to probabilities using the sigmoid function
probabilities = 1 / (1 + np.exp(-y_pred_test.predictions))

# Convert probabilities to binary predictions using a threshold of 0.5
threshold = 0.5
binary_predictions = (probabilities > threshold).astype(float)

# Now binary_predictions contains the binary predictions (0 or 1)
print(binary_predictions)


# In[ ]:


"""
Displaying confusion matrix for each label, took a little help from ChatGPT to
break it into group size of 3
"""

mcm = multilabel_confusion_matrix(list(test_df["labels"]), binary_predictions)

# Group labels into chunks of 3 for each figure
group_size = 3  # Number of boxes per figure
num_groups = (len(all_labels) + group_size - 1) // group_size  # Total number of groups

for group_idx in range(num_groups):
    fig, axes = plt.subplots(1, group_size, figsize=(21, 7), squeeze=False)  # 1 row, 3 columns
    axes = axes.ravel()

    for i in range(group_size):
        label_idx = group_idx * group_size + i
        if label_idx < len(all_labels):
            label = all_labels[label_idx]
            matrix = mcm[label_idx]

            # Plot the confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
            disp.plot(ax=axes[i], values_format='d')
            axes[i].set_title(f'Confusion Matrix for {label}', fontsize=18)
            axes[i].grid(False)

            # Adjust axis label sizes
            axes[i].tick_params(axis='x', labelsize=10)
            axes[i].tick_params(axis='y', labelsize=10)
            axes[i].xaxis.label.set_size(12)
            axes[i].yaxis.label.set_size(12)

            # Adjust the font size of the text annotations inside the confusion matrix
            for text in disp.ax_.texts:  # Access all annotation texts
                text.set_fontsize(16)  # Set font size to desired value
        else:
            # Hide unused subplot
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# In[ ]:


"""
Finding the misclassified examples and displaying them
"""

true_labels_int = np.array(test_df["labels"].tolist())

# Find indices of misclassified examples
misclassified_indices = np.where((true_labels_int != binary_predictions).any(axis=1))[0]

# Display misclassified examples
for idx in misclassified_indices:
    true_labels = ", ".join([all_labels[id] for id, i in enumerate(true_labels_int[idx]) if i == 1])
    predicted_labels = ", ".join([all_labels[id] for id, i in enumerate(binary_predictions[idx]) if i == 1])
    print(f"True Labels: {true_labels}")
    print(f"Predicted Labels: {predicted_labels}")
    print("------")

