import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, multilabel_confusion_matrix
from sklearn.pipeline import Pipeline
import re

class SVMBaseline:
    def __init__(self):
        self.labels = [
            "arts & culture", "business & entrepreneurs", "celebrity & pop culture",
            "diaries & daily life", "family", "fashion & style", "film tv & video",
            "fitness & health", "food & dining", "gaming",
            "learning & educational", "music", "news & social concern", "other hobbies",
            "relationships", "science & technology", "sports", "travel & adventure",
            "youth & student life"
        ]
        self.num_labels = len(self.labels)
        self.pipeline = None
    
    def _preprocess(self, text):
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"({URL})|@(user)", " ", text)
        text = text.lower()
        return text
    
    def _parse_labels(self, label_str):
        if pd.isna(label_str):
            return [0] * self.num_labels
        label_str = str(label_str)
        if not label_str.startswith('[') or not label_str.endswith(']'):
            return [0] * self.num_labels
        try:
            values = label_str.strip("[]").split()
            return list(map(int, values))
        except:
            return [0] * self.num_labels
    
    def load_data(self, train_path, val_path, test_path):
        train_df = pd.read_csv(train_path, encoding='latin-1')
        val_df = pd.read_csv(val_path, encoding='latin-1')
        test_df = pd.read_csv(test_path, encoding='latin-1')
        
        train_df["processed_text"] = train_df["text"].apply(self._preprocess)
        val_df["processed_text"] = val_df["text"].apply(self._preprocess)
        test_df["processed_text"] = test_df["text"].apply(self._preprocess)
        
        X_train = train_df["processed_text"].values
        y_train = np.array(train_df['gold_label_list'].apply(self._parse_labels).tolist())
        
        X_val = val_df["processed_text"].values
        y_val = np.array(val_df['gold_label_list'].apply(self._parse_labels).tolist())
        
        X_test = test_df["processed_text"].values
        y_test = np.array(test_df['gold_label_list'].apply(self._parse_labels).tolist())
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def build_model(self, kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('classifier', OneVsRestClassifier(
                SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    class_weight=class_weight,
                    probability=True,
                    random_state=42
                )
            ))
        ])
    
    def train(self, X_train, y_train):
        if self.pipeline is None:
            self.build_model()
        self.pipeline.fit(X_train, y_train)
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X, y, dataset_name=""):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1_micro = f1_score(y, y_pred, average='micro')
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_weighted = f1_score(y, y_pred, average='weighted')
        
        print(f"=== {dataset_name} Evaluation ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 (Micro): {f1_micro:.4f}")
        print(f"F1 (Macro): {f1_macro:.4f}")
        print(f"F1 (Weighted): {f1_weighted:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=self.labels))
        
        return {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }, y_pred

if __name__ == "__main__":
    svm = SVMBaseline()
    X_train, y_train, X_val, y_val, X_test, y_test = svm.load_data(
        'topic-train.csv',
        'topic-validation.csv',
        'topic-test.csv'
    )
    
    print(f"训练集: {X_train.shape[0]} samples")
    print(f"验证集: {X_val.shape[0]} samples")
    print(f"测试集: {X_test.shape[0]} samples")
    
    svm.build_model(kernel='rbf', C=10, gamma='scale')
    svm.train(X_train, y_train)
    
    print("\n" + "="*50)
    svm.evaluate(X_train, y_train, "训练集")
    
    print("\n" + "="*50)
    svm.evaluate(X_val, y_val, "验证集")
    
    print("\n" + "="*50)
    svm.evaluate(X_test, y_test, "测试集")