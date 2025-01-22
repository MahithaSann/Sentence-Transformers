import torch
from transformers import BertModel, BertTokenizer

class SentenceTransformer(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(SentenceTransformer, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, input_sentences):
        # Tokenize and encode sentences for BERT
        encoded_input = self.tokenizer(input_sentences, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(**encoded_input)
        # Use the pooler output (it's typically the output of the [CLS] token after passing through a dense layer)
        return outputs.pooler_output

# Initialize the model
model = SentenceTransformer()

sentences = ["Hello, how are you?", "Today is a sunny day.", "Deep learning transforms NLP."]

# Generate embeddings
embeddings = model(sentences)
print("Embeddings shape:", embeddings.shape)
print("embedding output:\n", embeddings)