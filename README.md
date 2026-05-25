文本主题分类项目
================

基于BERT和核SVM的多标签文本主题分类系统，用于对推文进行19个主题的分类。

项目结构
--------

```
Final_Code/
├── data_preprocessor.py    # 数据预处理模块
├── bert_multilabel_classifier.py  # BERT多标签分类器
├── svm_baseline.py         # SVM基线模型
├── optuna_hyperparameter_search.py  # Optuna超参搜索
├── trainer.py              # 训练器
├── visualization.py        # 可视化模块
├── main.py                 # 主脚本
├── topic-train.csv         # 训练数据
├── topic-validation.csv    # 验证数据
├── topic-test.csv          # 测试数据
└── README.md               # 项目说明
```

环境依赖
--------

需要安装以下Python库：

```bash
pip install pandas numpy scikit-learn transformers datasets torch optuna matplotlib seaborn
```

数据格式
--------

数据集包含推文文本和对应的19维标签向量：

- text: 推文内容
- date: 发布日期
- gold_label_list: 19维二进制标签列表，格式为 "[0 1 0 ... 0]"

19个主题标签：
1. arts & culture
2. business & entrepreneurs
3. celebrity & pop culture
4. diaries & daily life
5. family
6. fashion & style
7. film tv & video
8. fitness & health
9. food & dining
10. gaming
11. learning & educational
12. music
13. news & social concern
14. other hobbies
15. relationships
16. science & technology
17. sports
18. travel & adventure
19. youth & student life

运行方式
--------

### 运行完整流程

```bash
python main.py
```

### 单独运行SVM基线

```bash
python svm_baseline.py
```

### 单独运行BERT训练

```bash
python trainer.py
```

### 运行Optuna超参搜索

```bash
python optuna_hyperparameter_search.py
```

核心功能
--------

### 数据预处理
- 使用BERT tokenizer进行WordPiece分词
- 自动填充与截断，保证输入长度最多512 tokens
- 支持batched映射处理大规模数据

### BERT多标签分类器
- 重构输出层为19维sigmoid分类头
- 支持多标签文本分类（一条文本可对应多个主题）
- 集成Dropout正则化

### 核SVM基线
- 使用TF-IDF特征（max_features=5000, ngram_range=(1,2)）
- 采用OneVsRestClassifier + 核SVM

### 超参搜索
- 使用Optuna进行自动超参搜索
- 支持学习率、权重衰减、epoch、warmup_ratio等参数
- 采用TPESampler和MedianPruner

### 训练优化
- 自动batch size调整，从大到小查找可行batch
- 集成EarlyStopping，验证集F1不再提升时自动停止
- 支持FP16混合精度训练

### 可视化
- 标签相关性热图，定位语义重叠
- 混淆矩阵，定位少数类预测问题
- 模型性能对比图

性能指标
--------

| 模型 | 测试集F1(Weighted) |
|------|-------------------|
| SVM基线 | 0.4446 |
| BERT | 0.70+ |

BERT模型相较SVM基线提升约59%。

许可证
------

本项目仅供学习和研究使用。