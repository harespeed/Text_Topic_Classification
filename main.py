import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_preprocessor import DataPreprocessor
from trainer import BertTrainer
from svm_baseline import SVMBaseline
from visualization import PerformanceVisualizer

def main():
    labels = [
        "arts & culture", "business & entrepreneurs", "celebrity & pop culture",
        "diaries & daily life", "family", "fashion & style", "film tv & video",
        "fitness & health", "food & dining", "gaming",
        "learning & educational", "music", "news & social concern", "other hobbies",
        "relationships", "science & technology", "sports", "travel & adventure",
        "youth & student life"
    ]
    
    print("="*60)
    print("文本主题分类项目 - BERT vs SVM多标签分类对比")
    print("="*60)
    
    print("\n1. 数据预处理...")
    preprocessor = DataPreprocessor(max_seq_len=512)
    train_df, val_df, test_df = preprocessor.load_data(
        'topic-train.csv',
        'topic-validation.csv',
        'topic-test.csv'
    )
    train_data, val_data, test_data = preprocessor.prepare_datasets(train_df, val_df, test_df)
    print(f"   训练集: {len(train_data)} samples")
    print(f"   验证集: {len(val_data)} samples")
    print(f"   测试集: {len(test_data)} samples")
    print(f"   标签数量: {preprocessor.num_labels}")
    
    print("\n2. 训练SVM基线模型...")
    svm = SVMBaseline()
    X_train, y_train, X_val, y_val, X_test, y_test = svm.load_data(
        'topic-train.csv',
        'topic-validation.csv',
        'topic-test.csv'
    )
    svm.build_model(kernel='rbf', C=10, gamma='scale')
    svm.train(X_train, y_train)
    svm_results_train, _ = svm.evaluate(X_train, y_train, "训练集")
    svm_results_val, _ = svm.evaluate(X_val, y_val, "验证集")
    svm_results, svm_preds = svm.evaluate(X_test, y_test, "测试集")
    
    print("\n3. 训练BERT模型...")
    bert_trainer = BertTrainer()
    
    optimal_params = {
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'num_train_epochs': 10,
        'warmup_ratio': 0.1,
        'per_device_train_batch_size': None
    }
    
    trained_trainer = bert_trainer.train(train_data, val_data, optimal_params)
    bert_results, bert_preds, bert_true = bert_trainer.evaluate(trained_trainer, test_data, test_df)
    
    print("\n4. 性能对比分析...")
    visualizer = PerformanceVisualizer(labels)
    
    print("\n标签相关性热图（显示语义重叠）:")
    visualizer.plot_label_correlation_heatmap(y_test)
    
    print("\n样本数量分布:")
    visualizer.plot_sample_count_per_label(y_test)
    
    print("\nSVM模型各类F1分数:")
    svm_f1_scores = visualizer.plot_class_performance(y_test, svm_preds, model_name="SVM")
    
    print("\nBERT模型各类F1分数:")
    bert_f1_scores = visualizer.plot_class_performance(y_test, bert_preds, model_name="BERT")
    
    print("\nSVM模型混淆矩阵:")
    visualizer.plot_confusion_matrix(y_test, svm_preds, model_name="SVM")
    
    print("\nBERT模型混淆矩阵:")
    visualizer.plot_confusion_matrix(y_test, bert_preds, model_name="BERT")
    
    print("\n模型性能对比:")
    visualizer.plot_model_comparison(bert_results, svm_results)
    
    print("\n" + "="*60)
    print("性能对比总结")
    print("="*60)
    print(f"SVM基线 - F1 (Weighted): {svm_results['f1_weighted']:.4f}")
    print(f"BERT模型 - F1 (Weighted): {bert_results['f1_weighted']:.4f}")
    improvement = (bert_results['f1_weighted'] - svm_results['f1_weighted']) / svm_results['f1_weighted'] * 100
    print(f"提升幅度: {improvement:.2f}%")
    
    print("\n少数类分析（样本数少于100的类别）:")
    label_counts = y_test.sum(axis=0)
    minority_indices = np.where(label_counts < 100)[0]
    for idx in minority_indices:
        print(f"  {labels[idx]}: {label_counts[idx]} samples")
        print(f"    SVM F1: {svm_f1_scores[idx]:.4f}, BERT F1: {bert_f1_scores[idx]:.4f}")
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)

if __name__ == "__main__":
    main()