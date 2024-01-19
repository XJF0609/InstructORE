import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import random



class RelationshipDataset(Dataset):
    def __init__(self, sentences, entities, labels):
        self.sentences = sentences
        self.entities = entities
        self.labels = labels
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        
        sentence = self.sentences[idx]
        entity = self.entities[idx]
        label = self.labels[idx]
        
        
        input_text = f"{sentence} [E1]{entity[0]}[/E1] [E2]{entity[1]}[/E2]"
        
        inputs = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=128,  
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        
        return input_ids, attention_mask, label

class RelationshipModel(torch.nn.Module):
    def __init__(self):
        super(RelationshipModel, self).__init__()
        self.bert = bert_model
        self.fc = torch.nn.Linear(768, num_classes)  
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.fc(pooled_output)
        return logits
    
def demo_class():
    train_sentences = []
    train_entities = []
    train_labels = []
    with open() as sentence_1600:
        train_sentences = json.load(sentence_1600)
    with open() as entity_1600:
        train_entities = json.load(entity_1600)
    with open() as label_1600:
        train_labels = json.load(label_1600)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    train_dataset = RelationshipDataset(train_sentences, train_entities, train_labels)
    num_classes=16
    model = RelationshipModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    num_epochs = 2
    count=0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            count+=16
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)     
            optimizer.zero_grad()       
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)    
            loss.backward()
            optimizer.step()
            
        # model.eval()
    model_path=""
    torch.save(model,model_path)
    test_sentences = []
    test_entities = []
    with open("") as sentence_1600:
        test_sentences = random.shuffle(json.load(sentence_1600))[0:100]
    with open("") as entity_1600:
        test_entities = random.shuffle(json.load(entity_1600))[0:100]
    test_dataset = RelationshipDataset(test_sentences, test_entities, labels=None)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device) 
            logits = model(input_ids, attention_mask)
            probs = torch.nn.functional.softmax(logits, dim=1)
            _, predicted_labels = torch.max(probs, dim=1)
            predictions.extend(predicted_labels.cpu().numpy().tolist())