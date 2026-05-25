import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

class PerformanceVisualizer:
    def __init__(self, labels):
        self.labels = labels
        self.num_labels = len(labels)
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="", save_path=None):
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        
        group_size = 3
        num_groups = (self.num_labels + group_size - 1) // group_size
        
        for group_idx in range(num_groups):
            fig, axes = plt.subplots(1, group_size, figsize=(21, 7), squeeze=False)
            axes = axes.ravel()
            
            for i in range(group_size):
                label_idx = group_idx * group_size + i
                if label_idx < self.num_labels:
                    label = self.labels[label_idx]
                    matrix = mcm[label_idx]
                    
                    ax = axes[i]
                    sns.heatmap(matrix, annot=True, fmt='d', ax=ax, cmap='Blues')
                    ax.set_title(f'{model_name} - {label}', fontsize=14)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                else:
                    axes[i].axis('off')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_confusion_matrix_group_{group_idx}.png", dpi=100)
            plt.show()
    
    def plot_label_correlation_heatmap(self, y_true, y_pred=None, save_path=None):
        corr_matrix = np.corrcoef(y_true.T)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
                    xticklabels=self.labels, yticklabels=self.labels)
        plt.title('Label Correlation Heatmap', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_correlation_heatmap.png", dpi=100)
        plt.show()
    
    def plot_class_performance(self, y_true, y_pred, model_name="", save_path=None):
        from sklearn.metrics import f1_score
        
        f1_scores = []
        for i in range(self.num_labels):
            f1 = f1_score(y_true[:, i], y_pred[:, i], average='binary')
            f1_scores.append(f1)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x=self.labels, y=f1_scores, palette='viridis')
        plt.title(f'{model_name} - F1 Score per Class', fontsize=16)
        plt.ylabel('F1 Score')
        plt.xlabel('Labels')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_class_f1.png", dpi=100)
        plt.show()
        
        return f1_scores
    
    def plot_model_comparison(self, bert_results, svm_results, save_path=None):
        metrics = ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted']
        bert_vals = [bert_results[m] for m in metrics]
        svm_vals = [svm_results[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, bert_vals, width, label='BERT')
        rects2 = ax.bar(x + width/2, svm_vals, width, label='SVM')
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_comparison.png", dpi=100)
        plt.show()
    
    def plot_sample_count_per_label(self, y_true, save_path=None):
        counts = y_true.sum(axis=0)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x=self.labels, y=counts, palette='coolwarm')
        plt.title('Sample Count per Label', fontsize=16)
        plt.ylabel('Number of Samples')
        plt.xlabel('Labels')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_sample_counts.png", dpi=100)
        plt.show()
        
        return counts

if __name__ == "__main__":
    labels = [
        "arts & culture", "business & entrepreneurs", "celebrity & pop culture",
        "diaries & daily life", "family", "fashion & style", "film tv & video",
        "fitness & health", "food & dining", "gaming",
        "learning & educational", "music", "news & social concern", "other hobbies",
        "relationships", "science & technology", "sports", "travel & adventure",
        "youth & student life"
    ]
    
    visualizer = PerformanceVisualizer(labels)
    
    np.random.seed(42)
    y_true = np.random.randint(0, 2, (100, 19))
    y_pred = np.random.randint(0, 2, (100, 19))
    
    visualizer.plot_sample_count_per_label(y_true)
    visualizer.plot_label_correlation_heatmap(y_true)
    visualizer.plot_class_performance(y_true, y_pred, model_name="Test Model")