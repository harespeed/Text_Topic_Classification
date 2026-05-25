import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, multilabel_confusion_matrix
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoModelForSequenceClassification
from data_preprocessor import DataPreprocessor

class BertTrainer:
    def __init__(self, model_name="bert-base-uncased", num_labels=19):
        self.model_name = model_name
        self.num_labels = num_labels
        self.labels = [
            "arts & culture", "business & entrepreneurs", "celebrity & pop culture",
            "diaries & daily life", "family", "fashion & style", "film tv & video",
            "fitness & health", "food & dining", "gaming",
            "learning & educational", "music", "news & social concern", "other hobbies",
            "relationships", "science & technology", "sports", "travel & adventure",
            "youth & student life"
        ]
    
    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = (logits > 0.5).astype(int)
        acc = accuracy_score(labels, predictions)
        f1_micro = f1_score(labels, predictions, average='micro')
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        return {
            "accuracy": acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted
        }
    
    def find_optimal_batch_size(self, train_data):
        batch_sizes = [512, 256, 128, 64, 32, 16, 8]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for batch_size in batch_sizes:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=self.num_labels
                ).to(device)
                
                dummy_input = {
                    'input_ids': torch.randint(0, 30522, (batch_size, 512)).to(device),
                    'attention_mask': torch.ones((batch_size, 512)).to(device),
                    'labels': torch.rand((batch_size, self.num_labels)).to(device)
                }
                
                with torch.no_grad():
                    outputs = model(**dummy_input)
                
                print(f"Batch size {batch_size} is feasible")
                return batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Batch size {batch_size} causes OOM, trying smaller...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        return 8
    
    def train(self, train_data, val_data, params=None):
        if params is None:
            params = {
                'learning_rate': 2e-5,
                'weight_decay': 0.01,
                'num_train_epochs': 10,
                'warmup_ratio': 0.1,
                'per_device_train_batch_size': None
            }
        
        if params['per_device_train_batch_size'] is None:
            print("自动查找最佳batch size...")
            params['per_device_train_batch_size'] = self.find_optimal_batch_size(train_data)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )
        
        training_args = TrainingArguments(
            output_dir=f"./{self.model_name}-fine-tuned",
            num_train_epochs=params['num_train_epochs'],
            per_device_train_batch_size=params['per_device_train_batch_size'],
            per_device_eval_batch_size=params.get('per_device_eval_batch_size', 32),
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            warmup_ratio=params.get('warmup_ratio', 0.1),
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            report_to="none",
            seed=42,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=params.get('gradient_accumulation_steps', 1),
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        trainer.train()
        return trainer
    
    def evaluate(self, trainer, test_data, test_df):
        print("\n" + "="*60)
        print("在测试集上评估模型")
        print("="*60)
        
        predictions = trainer.predict(test_data)
        logits = predictions.predictions
        labels = test_df["labels"].tolist()
        
        binary_predictions = (logits > 0.5).astype(int)
        
        accuracy = accuracy_score(labels, binary_predictions)
        f1_micro = f1_score(labels, binary_predictions, average='micro')
        f1_macro = f1_score(labels, binary_predictions, average='macro')
        f1_weighted = f1_score(labels, binary_predictions, average='weighted')
        
        print(f"\n测试集评估结果:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 (Micro): {f1_micro:.4f}")
        print(f"F1 (Macro): {f1_macro:.4f}")
        print(f"F1 (Weighted): {f1_weighted:.4f}")
        
        print("\n分类报告:")
        print(classification_report(labels, binary_predictions, target_names=self.labels))
        
        return {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }, binary_predictions, np.array(labels)

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    train_df, val_df, test_df = preprocessor.load_data(
        'topic-train.csv',
        'topic-validation.csv',
        'topic-test.csv'
    )
    train_data, val_data, test_data = preprocessor.prepare_datasets(train_df, val_df, test_df)
    
    trainer = BertTrainer()
    trained_trainer = trainer.train(train_data, val_data)
    
    results, predictions, true_labels = trainer.evaluate(trained_trainer, test_data, test_df)
    
    print("\n训练完成!")
    print(f"测试集F1 (Weighted): {results['f1_weighted']:.4f}")