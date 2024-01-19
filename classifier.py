import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
import random
import numpy as np

class ClassifierNew:
    def __init__(self, k, sentences_file, max_seq_length, batch_size, num_epochs):
        with open(sentences_file) as f:
            self.data = f.readlines()
        self.k = k
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.tokenizer = self.create_tokenizer()
        self.model = self.build_model()
        self.device = self.assign_device()
        self.input_ids, self.masks = self.prepare_input()
        self.entity_indices = self.extract_entities()

    @staticmethod
    def assign_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device

    @staticmethod
    def create_tokenizer():
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
        special_tokens_dict = {'additional_special_tokens': [
            '[E1]', '[E2]', '[/E1]', '[/E2]']}  
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer

    def build_model(self):
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=self.k,
            output_attentions=False,
            output_hidden_states=True,
        )
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def prepare_input(self):
        input_ids = []
        masks = []
        for sent in self.data:
            encoded_dict = self.tokenizer.encode_plus(
                sent,                        
                add_special_tokens=True,     
                max_length=self.max_seq_length,     
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,  
                return_tensors='pt',         
            )
            input_ids.append(encoded_dict['input_ids'])
            masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        masks = torch.cat(masks, dim=0)
        print('Original: ', self.data[0])
        print('Token IDs:', input_ids[0])
        return input_ids, masks

    def extract_entities(self):
        e1_tks_id = self.tokenizer.convert_tokens_to_ids('[E1]')
        e2_tks_id = self.tokenizer.convert_tokens_to_ids('[E2]')
        entity_indices = []
        for input_id in self.input_ids:
            e1_idx = (input_id == e1_tks_id).nonzero().flatten().tolist()[0]
            e2_idx = (input_id == e2_tks_id).nonzero().flatten().tolist()[0]
            entity_indices.append((e1_idx, e2_idx))
        entity_indices = torch.Tensor(entity_indices)
        return entity_indices

    def get_hidden_features(self):
        self.model.to('cpu')
        outputs = self.model(self.input_ids, self.masks)
        return outputs[1][-1].detach().numpy().flatten().reshape(self.input_ids.shape[0], -1)

    def train_model(self, labels):
        labels = torch.tensor(labels).long()
        dataset = TensorDataset(self.input_ids, self.masks, labels)

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size])
        self.train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.batch_size
        )

        self.validation_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=self.batch_size
        )
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        epochs = self.num_epochs
        total_steps = len(self.train_dataloader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        self.model.cuda()
        for epoch_i in range(0, epochs):
            self.train_epoch()

    def train_epoch(self):
        total_train_loss = 0
        self.model.train()
        for batch in self.train_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            self.model.zero_grad()
            loss, logits, _ = self.model(b_input_ids,
                                         token_type_ids=None,
                                         attention_mask=b_input_mask,
                                         labels=b_labels)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
        avg_train_loss = total_train_loss / len(self.train_dataloader)
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in self.validation_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            with torch.no_grad():
                (loss, logits, _) = self.model(b_input_ids,
                                               token_type_ids=None,
                                               attention_mask=b_input_mask,
                                               labels=b_labels)
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += self.calculate_accuracy(logits, label_ids)
        avg_val_accuracy = total_eval_accuracy / \
            len(self.validation_dataloader)
        avg_val_loss = total_eval_loss / len(self.validation_dataloader)