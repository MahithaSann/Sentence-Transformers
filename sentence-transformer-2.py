import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=3, num_sentiments=2):
        super(MultiTaskSentenceTransformer, self).__init__()
        # Load the BERT model as the shared encoder
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Task-specific heads
        # Classification head: Assume 3 classes (e.g., Science, Sports, Technology)
        self.classification_head = nn.Linear(self.bert.config.hidden_size, num_classes)
        # Sentiment analysis head: Assume binary sentiment (positive, negative)
        self.sentiment_head = nn.Linear(self.bert.config.hidden_size, num_sentiments)

    def forward(self, input_sentences):
        # Encoding sentences
        encoded_input = self.tokenizer(input_sentences, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(**encoded_input)
        
        # Using the [CLS] token to represent the whole sentence's embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Task-specific outputs
        classification_output = self.classification_head(cls_embeddings)
        sentiment_output = self.sentiment_head(cls_embeddings)
        
        return classification_output, sentiment_output

# Example usage
model = MultiTaskSentenceTransformer()

# Expanded list of sample input sentences
sentences = [
    "The discovery of exoplanets has expanded our understanding of the universe.",
    "The championship game will be held this weekend at the stadium.",
    "Technological advances are shaping the future of our societies.",
    "The political debate last night was quite intense and heated.",
    "She loves the new phone because it takes amazing photos.",
    "He dislikes the cold weather of the winter months.",
    "Artificial Intelligence is driving innovation in multiple sectors.",
    "Global warming poses a significant threat to world ecosystems."
]

# Forward pass (dummy example, no actual training or output interpretation)
classification_output, sentiment_output = model(sentences)
print("Classification Output:", classification_output)
print("Sentiment Output:", sentiment_output)
