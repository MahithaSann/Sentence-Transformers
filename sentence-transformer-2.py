import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-large-uncased', num_classes=3, num_sentiments=2):
        super(MultiTaskSentenceTransformer, self).__init__()
        # Load BERT model 
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # More sophisticated task-specific heads
        self.classification_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
        
        self.sentiment_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_sentiments)
        )

    def forward(self, input_sentences):
        # Encoding sentences
        encoded_input = self.tokenizer(
            input_sentences, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        outputs = self.bert(**encoded_input)
        
        # Using the [CLS] token to represent the whole sentence's embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Task-specific outputs
        classification_output = self.classification_head(cls_embeddings)
        sentiment_output = self.sentiment_head(cls_embeddings)
        
        return classification_output, sentiment_output

# Classification and sentiment labels
classification_labels = ["Science", "Sports", "Technology"]
sentiment_labels = ["Negative", "Positive"]

# Sample input sentences
sentences = [
    "The championship game will be held this weekend at the stadium.",
    "Technological advances are shaping the future of our societies.",
    "The political debate last night was quite intense and heated.",
    "She loves the new phone because it takes amazing photos.",
    "He dislikes the cold weather of the winter months.",
    "Global warming poses a significant threat to world ecosystems."
]

# Model initialization
model = MultiTaskSentenceTransformer()

# Inference
with torch.no_grad():
    # Forward pass
    classification_output, sentiment_output = model(sentences)
    
    # Apply softmax to get probabilities
    classification_probs = F.softmax(classification_output, dim=1)
    sentiment_probs = F.softmax(sentiment_output, dim=1)
    
    # Get predicted classes and sentiments
    classification_predictions = [
        classification_labels[torch.argmax(prob)] 
        for prob in classification_probs
    ]
    sentiment_predictions = [
        sentiment_labels[torch.argmax(prob)] 
        for prob in sentiment_probs
    ]

print("Detailed Predictions:\n")
for sentence, class_pred, sent_pred, class_prob, sent_prob in zip(
    sentences, 
    classification_predictions, 
    sentiment_predictions,
    classification_probs,
    sentiment_probs
):
    print(f"Sentence: '{sentence}'")
    print(f"  Predicted Class: {class_pred}")
    print(f"  Class Probabilities: {[f'{p:.2f}' for p in class_prob.tolist()]}")
    print(f"  Predicted Sentiment: {sent_pred}")
    print(f"  Sentiment Probabilities: {[f'{p:.2f}' for p in sent_prob.tolist()]}")
    print("-----")