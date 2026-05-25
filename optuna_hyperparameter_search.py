import optuna
import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from data_preprocessor import DataPreprocessor
from bert_multilabel_classifier import BertMultilabelClassifier

class OptunaHyperparameterSearch:
    def __init__(self, train_data, val_data, num_labels=19):
        self.train_data = train_data
        self.val_data = val_data
        self.num_labels = num_labels
        self.best_params = None
        self.best_score = 0.0
    
    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = (logits > 0.5).astype(int)
        f1 = f1_score(labels, predictions, average='weighted')
        return {"f1": f1}
    
    def objective(self, trial):
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        num_train_epochs = trial.suggest_int('num_train_epochs', 3, 10)
        warmup_ratio = trial.suggest_uniform('warmup_ratio', 0.0, 0.2)
        per_device_train_batch_size = trial.suggest_categorical(
            'per_device_train_batch_size', [8, 16, 32]
        )
        
        model = BertMultilabelClassifier(
            model_name="bert-base-uncased",
            num_labels=self.num_labels,
            dropout_rate=0.1
        )
        
        training_args = TrainingArguments(
            output_dir=f"./optuna_trial_{trial.number}",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=32,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
            seed=42,
            fp16=torch.cuda.is_available(),
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        trainer.train()
        eval_results = trainer.evaluate()
        f1_score_val = eval_results.get('eval_f1', 0.0)
        
        if f1_score_val > self.best_score:
            self.best_score = f1_score_val
            self.best_params = {
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'num_train_epochs': num_train_epochs,
                'warmup_ratio': warmup_ratio,
                'per_device_train_batch_size': per_device_train_batch_size
            }
        
        return f1_score_val
    
    def search(self, n_trials=20):
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=1
        )
        
        print("\n" + "="*50)
        print("最佳超参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"\n最佳F1分数: {study.best_value:.4f}")
        
        return study.best_params, study.best_value

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    train_df, val_df, test_df = preprocessor.load_data(
        'topic-train.csv',
        'topic-validation.csv',
        'topic-test.csv'
    )
    train_data, val_data, test_data = preprocessor.prepare_datasets(train_df, val_df, test_df)
    
    searcher = OptunaHyperparameterSearch(train_data, val_data)
    best_params, best_score = searcher.search(n_trials=5)
    
    print("\n搜索完成!")
    print(f"最佳参数: {best_params}")
    print(f"最佳验证F1: {best_score:.4f}")