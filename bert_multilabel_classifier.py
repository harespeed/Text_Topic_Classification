import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BertMultilabelClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=19, dropout_rate=0.1):
        super(BertMultilabelClassifier, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        cls_output = outputs[1]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        probabilities = self.sigmoid(logits)
        
        loss = None
        if labels is not None:
            loss_fn = nn.BCELoss()
            loss = loss_fn(probabilities, labels)
        
        return (loss, probabilities) if loss is not None else probabilities

if __name__ == "__main__":
    model = BertMultilabelClassifier()
    print(model)
    
    # Test forward pass
    input_ids = torch.randint(0, 30522, (2, 512))
    attention_mask = torch.ones((2, 512))
    labels = torch.rand((2, 19))
    
    loss, probs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(f"Loss: {loss.item()}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")