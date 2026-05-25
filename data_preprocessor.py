import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

class DataPreprocessor:
    def __init__(self, model_name="bert-base-uncased", max_seq_len=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_len = max_seq_len
        self.labels = [
            "arts & culture", "business & entrepreneurs", "celebrity & pop culture",
            "diaries & daily life", "family", "fashion & style", "film tv & video",
            "fitness & health", "food & dining", "gaming",
            "learning & educational", "music", "news & social concern", "other hobbies",
            "relationships", "science & technology", "sports", "travel & adventure",
            "youth & student life"
        ]
        self.num_labels = len(self.labels)
    
    def load_data(self, train_path, val_path, test_path):
        train_df = pd.read_csv(train_path, encoding='latin-1')
        val_df = pd.read_csv(val_path, encoding='latin-1')
        test_df = pd.read_csv(test_path, encoding='latin-1')
        
        train_df["labels"] = train_df['gold_label_list'].apply(self._parse_labels)
        val_df["labels"] = val_df['gold_label_list'].apply(self._parse_labels)
        test_df["labels"] = test_df['gold_label_list'].apply(self._parse_labels)
        
        return train_df, val_df, test_df
    
    def _parse_labels(self, label_str):
        if pd.isna(label_str):
            return [0.0] * self.num_labels
        label_str = str(label_str)
        if not label_str.startswith('[') or not label_str.endswith(']'):
            return [0.0] * self.num_labels
        try:
            values = label_str.strip("[]").split()
            return list(map(float, values))
        except:
            return [0.0] * self.num_labels
    
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_overflowing_tokens=False
        )
    
    def prepare_datasets(self, train_df, val_df, test_df):
        train_data = Dataset.from_pandas(train_df)
        val_data = Dataset.from_pandas(val_df)
        test_data = Dataset.from_pandas(test_df)
        
        tokenized_train = train_data.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text', 'date', 'gold_label_list']
        )
        tokenized_val = val_data.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text', 'date', 'gold_label_list']
        )
        tokenized_test = test_data.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text', 'date', 'gold_label_list']
        )
        
        tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return tokenized_train, tokenized_val, tokenized_test

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    train_df, val_df, test_df = preprocessor.load_data(
        'topic-train.csv',
        'topic-validation.csv',
        'topic-test.csv'
    )
    train_data, val_data, test_data = preprocessor.prepare_datasets(train_df, val_df, test_df)
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    print(f"测试集样本数: {len(test_data)}")
    print(f"特征维度: {train_data[0]['input_ids'].shape}")
    print(f"标签数量: {preprocessor.num_labels}")