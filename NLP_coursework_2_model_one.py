# -*- coding: utf-8 -*-
# Auto-generated from Jupyter notebook: NLP_coursework_2_model_one.ipynb
# Markdown cells are converted to comments below.


# %% [markdown] cell 1
# # Machine Learning for NLP COURSEWORK 2 model one
# ## Make sure to add all training, validation and testing dataset to the root and word embeddings file 'glove.6B.300d.txt' in the root
# 
# ## Group members:
# ### - Naeem Ali
# ### - Qiang Fu
# ### - Qiran Sun
# ### - Yanzong Qu
# ### - Zhaoliang Liu

# %% [markdown] cell 2
# # Libraries

# %% [code] cell 3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import re
from sklearn.metrics import classification_report, accuracy_score, f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
nltk.download('punkt_tab')

# %% [markdown] cell 4
# # Loading dataset

# %% [code] cell 5
"""
Reading training, validation and testing dataset
"""
data_train = pd.read_csv('topic-train.csv')
data_valid = pd.read_csv('topic-validation.csv')
data_test = pd.read_csv('topic-test.csv')

"""
Combining all datasets into one for EDA
"""
all_data = pd.concat([data_train, data_valid, data_test], ignore_index=True)

"""
Converting string representation of gold labels to integer list
"""
all_data['gold_label_list'] = all_data['gold_label_list'].apply(lambda x: list(map(int, x.strip("[]").split())))
data_train['gold_label_list'] = data_train['gold_label_list'].apply(lambda x: list(map(int, x.strip("[]").split())))
data_valid['gold_label_list'] = data_valid['gold_label_list'].apply(lambda x: list(map(int, x.strip("[]").split())))
data_test['gold_label_list'] = data_test['gold_label_list'].apply(lambda x: list(map(int, x.strip("[]").split())))

# %% [markdown] cell 6
# # EDA

# %% [code] cell 7
print("Combined Dataset: ", all_data.shape)
print("Training dataset: ", data_train.shape)
print("Validation dataset: ", data_valid.shape)
print("Testing dataset: ", data_test.shape)

# %% [code] cell 8
"""
List of labels in the dataset
"""
all_labels = ["arts & culture", "business & entrepreneurs", "celebrity & pop culture",
              "diaries & daily life", "family", "fashion & style", "film tv & video",
              "fitness & health", "food & dining", "gaming",
              "learning & educational", "music", "news & social concern", "other hobbies",
              "relationships", "science & technology", "sports", "travel & adventure",
              "youth & student life"]

"""
Spreading gold labels into separate columns for easy calculations for charts
"""
all_data_gold_labels_df = pd.DataFrame(all_data['gold_label_list'].tolist(), columns=all_labels)
train_data_gold_labels_df = pd.DataFrame(data_train['gold_label_list'].tolist(), columns=all_labels)
valid_data_gold_labels_df = pd.DataFrame(data_valid['gold_label_list'].tolist(), columns=all_labels)
test_data_gold_labels_df = pd.DataFrame(data_test['gold_label_list'].tolist(), columns=all_labels)

# %% [code] cell 9
"""
Creating three bar charts for our combined dataset.
First, shows number of tweets per topic
Second, shows number of labels per tweet
Third, shows time period of tweets in dataset
"""

categories = np.array(all_data_gold_labels_df.columns)
data_to_plot = all_data_gold_labels_df.iloc[:, :].sum().values

df_to_plot = pd.DataFrame({'categories': categories, 'values': data_to_plot})

sns.set(font_scale=2)
plt.figure(figsize=(15, 8))

ax = sns.barplot(x='categories', y='values', data=df_to_plot)

plt.title("Topic Classification - Combined Dataset", fontsize=24)
plt.ylabel('Number of tweets per topic', fontsize=18)
plt.xlabel('Topics ', fontsize=18)

plt.xticks(rotation=45, ha='right') # Rotate labels by 45 degrees and align horizontally
plt.show()

data_to_plot = all_data_gold_labels_df.iloc[:, :].sum(axis=1)
data_to_plot = data_to_plot.value_counts()
data_to_plot = data_to_plot.iloc[:]

# Create a DataFrame for plotting
df_to_plot = pd.DataFrame({'categories': data_to_plot.index, 'values': data_to_plot.values})

sns.set(font_scale=2)
plt.figure(figsize=(15, 8))

# Use the 'x' and 'y' parameters to specify columns within the DataFrame
ax = sns.barplot(x='categories', y='values', data=df_to_plot)

plt.title("Topic Classification - Combined Dataset", fontsize=24)
plt.ylabel('Number of labels per tweet', fontsize=18)
plt.xlabel('Topics ', fontsize=18)

rects = ax.patches
for rect, label in zip(rects, data_to_plot.values):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)
plt.show()

# Convert the date column to datetime objects if it's not already
all_data['date'] = pd.to_datetime(all_data['date'], format="mixed")

# Group the data by date and count the number of tweets for each date
date_counts = all_data.groupby('date').size().reset_index(name='counts')

# Create the plot
plt.figure(figsize=(15, 6))
sns.lineplot(x='date', y='counts', data=date_counts)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Tweets', fontsize=12)
plt.title('Number of Tweets Over Time - Combined Dataset', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [code] cell 10
"""
Creating three bar charts for our training dataset.
First, shows number of tweets per topic
Second, shows number of labels per tweet
Third, shows time period of tweets in dataset
"""

categories = np.array(train_data_gold_labels_df.columns)
data_to_plot = train_data_gold_labels_df.iloc[:, :].sum().values

df_to_plot = pd.DataFrame({'categories': categories, 'values': data_to_plot})

sns.set(font_scale=2)
plt.figure(figsize=(15, 8))

ax = sns.barplot(x='categories', y='values', data=df_to_plot)

plt.title("Topic Classification - Train Dataset", fontsize=24)
plt.ylabel('Number of tweets per topic', fontsize=18)
plt.xlabel('Topics ', fontsize=18)

plt.xticks(rotation=45, ha='right') # Rotate labels by 45 degrees and align horizontally
plt.show()

data_to_plot = train_data_gold_labels_df.iloc[:, :].sum(axis=1)
data_to_plot = data_to_plot.value_counts()
data_to_plot = data_to_plot.iloc[:]

# Create a DataFrame for plotting
df_to_plot = pd.DataFrame({'categories': data_to_plot.index, 'values': data_to_plot.values})

sns.set(font_scale=2)
plt.figure(figsize=(15, 8))

# Use the 'x' and 'y' parameters to specify columns within the DataFrame
ax = sns.barplot(x='categories', y='values', data=df_to_plot)

plt.title("Topic Classification - Train Dataset", fontsize=24)
plt.ylabel('Number of labels per tweet', fontsize=18)
plt.xlabel('Topics ', fontsize=18)

rects = ax.patches
for rect, label in zip(rects, data_to_plot.values):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)
plt.show()

# Convert the date column to datetime objects if it's not already
data_train['date'] = pd.to_datetime(data_train['date'], format="mixed")

# Group the data by date and count the number of tweets for each date
date_counts = data_train.groupby('date').size().reset_index(name='counts')

# Create the plot
plt.figure(figsize=(15, 6))
sns.lineplot(x='date', y='counts', data=date_counts)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Tweets', fontsize=12)
plt.title('Number of Tweets Over Time - Training Dataset', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [code] cell 11
"""
Creating three bar charts for our validation dataset.
First, shows number of tweets per topic
Second, shows number of labels per tweet
Third, shows time period of tweets in dataset
"""

categories = np.array(valid_data_gold_labels_df.columns)
data_to_plot = valid_data_gold_labels_df.iloc[:, :].sum().values

df_to_plot = pd.DataFrame({'categories': categories, 'values': data_to_plot})

sns.set(font_scale=2)
plt.figure(figsize=(15, 8))

# Use the 'x' and 'y' parameters to specify columns within the DataFrame
ax = sns.barplot(x='categories', y='values', data=df_to_plot)

plt.title("Topic Classification - Validation Dataset", fontsize=24)
plt.ylabel('Number of tweets per topic', fontsize=18)
plt.xlabel('Topics ', fontsize=18)

plt.xticks(rotation=45, ha='right') # Rotate labels by 45 degrees and align horizontally
plt.show()

data_to_plot = valid_data_gold_labels_df.iloc[:, :].sum(axis=1)
data_to_plot = data_to_plot.value_counts()
data_to_plot = data_to_plot.iloc[:]
# Create a DataFrame for plotting
df_to_plot = pd.DataFrame({'categories': data_to_plot.index, 'values': data_to_plot.values})

sns.set(font_scale=2)
plt.figure(figsize=(15, 8))

# Use the 'x' and 'y' parameters to specify columns within the DataFrame
ax = sns.barplot(x='categories', y='values', data=df_to_plot)

plt.title("Topic Classification - Validation Dataset", fontsize=24)
plt.ylabel('Number of tweets per topic', fontsize=18)
plt.xlabel('Topics ', fontsize=18)

rects = ax.patches
for rect, label in zip(rects, data_to_plot.values):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)
plt.show()


# Convert the date column to datetime objects if it's not already
data_valid['date'] = pd.to_datetime(data_valid['date'], format="mixed")

# Group the data by date and count the number of tweets for each date
date_counts = data_valid.groupby('date').size().reset_index(name='counts')

# Create the plot
plt.figure(figsize=(15, 6))
sns.lineplot(x='date', y='counts', data=date_counts)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Tweets', fontsize=12)
plt.title('Number of Tweets Over Time - Validation Dataset', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [code] cell 12
"""
Creating three bar charts for our testing dataset.
First, shows number of tweets per topic
Second, shows number of labels per tweet
Third, shows time period of tweets in dataset
"""

categories = np.array(test_data_gold_labels_df.columns)
data_to_plot = test_data_gold_labels_df.iloc[:, :].sum().values

df_to_plot = pd.DataFrame({'categories': categories, 'values': data_to_plot})

sns.set(font_scale=2)
plt.figure(figsize=(15, 8))

# Use the 'x' and 'y' parameters to specify columns within the DataFrame
ax = sns.barplot(x='categories', y='values', data=df_to_plot)

plt.title("Topic Classification - Testing Dataset", fontsize=24)
plt.ylabel('Number of tweets per topic', fontsize=18)
plt.xlabel('Topics ', fontsize=18)

plt.xticks(rotation=45, ha='right') # Rotate labels by 45 degrees and align horizontally
plt.show()

data_to_plot = test_data_gold_labels_df.iloc[:, :].sum(axis=1)
data_to_plot = data_to_plot.value_counts()
data_to_plot = data_to_plot.iloc[:]
# Create a DataFrame for plotting
df_to_plot = pd.DataFrame({'categories': data_to_plot.index, 'values': data_to_plot.values})

sns.set(font_scale=2)
plt.figure(figsize=(15, 8))

# Use the 'x' and 'y' parameters to specify columns within the DataFrame
ax = sns.barplot(x='categories', y='values', data=df_to_plot)

plt.title("Topic Classification - Testing Dataset", fontsize=24)
plt.ylabel('Number of tweets per topic', fontsize=18)
plt.xlabel('Topics ', fontsize=18)

rects = ax.patches
for rect, label in zip(rects, data_to_plot.values):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)
plt.show()

# Convert the date column to datetime objects if it's not already
data_test['date'] = pd.to_datetime(data_test['date'], format="mixed")

# Group the data by date and count the number of tweets for each date
date_counts = data_test.groupby('date').size().reset_index(name='counts')

# Create the plot
plt.figure(figsize=(15, 6))
sns.lineplot(x='date', y='counts', data=date_counts)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Tweets', fontsize=12)
plt.title('Number of Tweets Over Time - Testing Dataset', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [code] cell 13
average_labels_per_tweet_combined = all_data_gold_labels_df.sum(axis=1).mean()
print(f"The average number of labels per tweet across the combined dataset is: {average_labels_per_tweet_combined}")

average_labels_per_tweet_training = train_data_gold_labels_df.sum(axis=1).mean()
print(f"The average number of labels per tweet across the training dataset is: {average_labels_per_tweet_training}")

average_labels_per_tweet_validation = valid_data_gold_labels_df.sum(axis=1).mean()
print(f"The average number of labels per tweet across the validation dataset is: {average_labels_per_tweet_validation}")

average_labels_per_tweet_testing = test_data_gold_labels_df.sum(axis=1).mean()
print(f"The average number of labels per tweet across the testing dataset is: {average_labels_per_tweet_testing}")

# %% [code] cell 14
categories = np.array(test_data_gold_labels_df.columns)
data_to_plot = test_data_gold_labels_df.iloc[:, :].sum().values

df_to_plot = pd.DataFrame({'categories': categories, 'values': data_to_plot})

most_common = data_to_plot.max()
most_least = data_to_plot.min()

colors = ['green' if data == most_common else ('red' if data == most_least else 'blue') for data in data_to_plot]

sns.set(font_scale=2)
plt.figure(figsize=(15, 8))

# Use the 'x' and 'y' parameters to specify columns within the DataFrame
ax = sns.barplot(x='categories', y='values', data=df_to_plot, palette=colors)

plt.title("Most common and least number of topics in test dataset", fontsize=24)
plt.ylabel('Number of tweets per topic', fontsize=18)
plt.xlabel('Topics ', fontsize=18)

plt.xticks(rotation=45, ha='right') # Rotate labels by 45 degrees and align horizontally
plt.show()

# %% [code] cell 15
categories = np.array(train_data_gold_labels_df.columns)
data_to_plot = train_data_gold_labels_df.iloc[:, :].sum().values

df_to_plot = pd.DataFrame({'categories': categories, 'values': data_to_plot})

most_common = data_to_plot.max()
most_least = data_to_plot.min()

colors = ['green' if data == most_common else ('red' if data == most_least else 'blue') for data in data_to_plot]

sns.set(font_scale=2)
plt.figure(figsize=(15, 8))

# Use the 'x' and 'y' parameters to specify columns within the DataFrame
ax = sns.barplot(x='categories', y='values', data=df_to_plot, palette=colors)

plt.title("Most common and least number of topics in train dataset", fontsize=24)
plt.ylabel('Number of tweets per topic', fontsize=18)
plt.xlabel('Topics ', fontsize=18)

plt.xticks(rotation=45, ha='right') # Rotate labels by 45 degrees and align horizontally
plt.show()

# %% [code] cell 16
categories = np.array(valid_data_gold_labels_df.columns)
data_to_plot = valid_data_gold_labels_df.iloc[:, :].sum().values

df_to_plot = pd.DataFrame({'categories': categories, 'values': data_to_plot})

most_common = data_to_plot.max()
most_least = data_to_plot.min()

colors = ['green' if data == most_common else ('red' if data == most_least else 'blue') for data in data_to_plot]

sns.set(font_scale=2)
plt.figure(figsize=(15, 8))

# Use the 'x' and 'y' parameters to specify columns within the DataFrame
ax = sns.barplot(x='categories', y='values', data=df_to_plot, palette=colors)

plt.title("Most common and least number of topics in validation dataset", fontsize=24)
plt.ylabel('Number of tweets per topic', fontsize=18)
plt.xlabel('Topics ', fontsize=18)

plt.xticks(rotation=45, ha='right') # Rotate labels by 45 degrees and align horizontally
plt.show()

# %% [markdown] cell 17
# # Loading word embedding

# %% [code] cell 18
"""
This function will load the GloVe embeddings into a dictionary.
"""
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Path to GloVe embeddings
word_vectors = load_glove_embeddings('glove.6B.300d.txt')

# %% [markdown] cell 19
# # Helper methods

# %% [code] cell 20
"""
Preprocessing function to remove non-alphabetical, words like URL and @user,
and finally lowercasing the sentence
"""
def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove non-alphabetical characters
    text = re.sub(r"({URL})|@(user)", " ", text)
    text = text.lower()  # convert to lowercase
    return text

"""
This method will get the average word embedding for a sentence.
"""
def get_document_vector(text, word_vectors):
    tokens = word_tokenize(text)
    # Get the word vectors and average them
    vectors = [word_vectors[word] for word in tokens if word in word_vectors]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word_vectors.vector_size)  # Return zero vector if no word found

# %% [markdown] cell 21
# # Preprocessing datasets

# %% [code] cell 22
"""
preprocessing our datasets.
"""
data_train["processed_text"] = data_train["text"].apply(preprocess)
X_train = np.array([get_document_vector(text, word_vectors) for text in data_train['processed_text']])
y_train = np.array(list(data_train['gold_label_list']))

data_valid["processed_text"] = data_valid["text"].apply(preprocess)
X_valid = np.array([get_document_vector(text, word_vectors) for text in data_valid['processed_text']])
y_valid = np.array(list(data_valid['gold_label_list']))

data_test["processed_text"] = data_test["text"].apply(preprocess)
X_test = np.array([get_document_vector(text, word_vectors) for text in data_test['processed_text']])
y_test = np.array(list(data_test['gold_label_list']))

# %% [markdown] cell 23
# # Training model

# %% [code] cell 24
"""
Change the run_grid_search to True, to run extensive search for best parameter.
I am setting this variable to False because it takes a lot of time to run and I don't have the resources.
"""
run_grid_search = False

if run_grid_search:
  param_grid = {
      'estimator__kernel': ['linear', 'rbf', 'poly'],
      'estimator__C': [0.1, 1, 10],
      'estimator__gamma': [0.1, 1, 'scale', 'auto'],
      'estimator__class_weight': ['balanced', None]
  }
  """
  Using RandomizedSearchCV to find the best parameters which searches for some of the best, otherwise we can also use
  GridSearchCV to find the best parameters which will run exhaustive search on all the parameters.
  """
  grid_search = RandomizedSearchCV(OneVsRestClassifier(SVC()), param_grid, cv=3, verbose=2, n_iter=4)
  grid_search.fit(X_train, y_train)
  best_model = grid_search.best_estimator_
  print("Best model:", best_model)
else:
  """
  using some best parameters, found using random search when implementing.
  """
  clf = OneVsRestClassifier(SVC(kernel='linear', C=10))
  clf.fit(X_train, y_train)
  best_model = clf

# %% [markdown] cell 25
# # Model accuracy

# %% [code] cell 26
"""
Using our validation dataset to evaluate our model ensuring our dataset has not
seen this data in training.
"""
y_pred_valid = best_model.predict(X_valid)
print(f'Accuracy: {accuracy_score(y_valid, y_pred_valid)}')
print(f'F1-score: {f1_score(y_valid, y_pred_valid, average="weighted")}')
print("Validation classification report")
print(classification_report(y_valid, y_pred_valid))

"""
Finally evaluating our model on the test dataset
"""
y_pred_test = best_model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred_test)}')
print(f'F1-score: {f1_score(y_test, y_pred_test, average="weighted")}')
print("Test classification report")
print(classification_report(y_test, y_pred_test))

# %% [markdown] cell 27
# # Error Analysis

# %% [code] cell 28
"""
Printing confusion matrix for each label, took a little help of AI for this.
"""

from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay

mcm = multilabel_confusion_matrix(y_test, y_pred_test)

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

# %% [code] cell 29
"""
Finding the misclassified tweet that our model has classified.
"""
misclassified_indices = np.where((y_test != y_pred_test).any(axis=1))[0]

# Display misclassified examples
for idx in misclassified_indices:
    true_labels = ", ".join([all_labels[id] for id, i in enumerate(y_test[idx]) if i == 1])
    predicted_labels = ", ".join([all_labels[id] for id, i in enumerate(y_pred_test[idx]) if i == 1])
    print(f"True Labels: {true_labels}")
    print(f"Predicted Labels: {predicted_labels}")
    print("------")
