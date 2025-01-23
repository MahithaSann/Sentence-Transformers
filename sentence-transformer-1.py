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
embeddings = model(sentences)
print("Embeddings shape:", embeddings.shape)
print("embedding output:\n", embeddings)

"""
Discussion:

Architectural Choices
Transformer Backbone: I chose bert-base-uncased because it's a versatile model that provides good contextual embeddings across a variety of domains. 
It's also not as large as other variants like BERT-large, making it a reasonable choice in terms of computational efficiency.

Output Embeddings: Using the pooler_output which is typically designed to capture the essence of the input sequence in tasks that require understanding the whole input as one entity 
(like classification tasks). This output comes from the final hidden state of the [CLS] token, which is why it's suitable for our purpose of getting a fixed-length embedding.

Tokenization and Input Formatting: The choice to handle padding and truncation directly in the model allows us to abstract these preprocessing steps 
from the end-user, making the model more flexible with varying lengths of input sentences.

Conclusion:
This implementation provides a robust method to convert sentences into embeddings that can be used for 
various applications such as sentence similarity, clustering, or as features in downstream machine learning models. 
The architecture is kept simple to focus on producing reliable embeddings. In real-world applications, 
you might consider fine-tuning on specific tasks or datasets, or exploring other embedding strategies like averaging all token embeddings 
instead of using just the [CLS] token for potentially richer sentence representations.


"""