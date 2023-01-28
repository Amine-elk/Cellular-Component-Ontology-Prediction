# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 20:27:39 2023

@author: elfak
"""
import os 
os.chdir('C:/Users/elfak/OneDrive/Bureau/altegrad_challenge_2022')

import numpy as np
import networkx as nx
import sklearn
from sklearn import preprocessing

import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import csv
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import sentencepiece as spm

import numpy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Read sequences
sequences = list()
with open('data/sequences.txt', 'r', encoding="utf-8") as f:
    for line in f:
        sequences.append(line[:-1])

# Split data into training and test sets
sequences_train = list()
sequences_test = list()
proteins_test = list()
y_train = list()
with open('data/graph_labels.txt', 'r') as f:
    for i,line in enumerate(f):
        t = line.split(',')
        if len(t[1][:-1]) == 0:
            proteins_test.append(t[0])
            sequences_test.append(sequences[i])
        else:
            sequences_train.append(sequences[i])
            y_train.append(int(t[1][:-1]))

class TransformerModel(nn.Module):
    def __init__(self, ntoken, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        '''
        ntokens: the size of vocabulary
        nhid: the hidden dimension of the model.
        We assume that embedding_dim = nhid
        nlayers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead: the number of heads in the multiheadattention models
        dropout: the dropout value
         '''
        self.model_type = "Transformer"
        self.encoder = nn.Embedding(ntoken,nhid) # fill me, nhid = the dim_embed
        self.pos_encoder = PositionalEncoding(nhid,dropout) #fill me, the PositionalEncoding class is implemented in the next cell
        encoder_layers = nn.TransformerEncoderLayer(nhid,nhead,nhid,dropout) #fill me we assume nhid = d_model = dim_feedforward
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers) #fill me
        self.nhid = nhid
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.nhid) 
        src = self.pos_encoder(src) #fill me
        output = self.transformer_encoder(src, src_mask) #fill me
        return output


class ClassificationHead(nn.Module):
    def __init__(self, nhid, nclasses):
        super(ClassificationHead, self).__init__()
        self.decoder = nn.Linear(nhid, nclasses)  #fill me
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        output = self.decoder(src)
        return output
    
class Model(nn.Module):
    def __init__(self, ntoken, nhead, nhid, nlayers, nclasses, dropout=0.5):
        super(Model, self).__init__() 
        self.base = TransformerModel(ntoken, nhead, nhid, nlayers, dropout) #fill me
        self.classifier = ClassificationHead(nhid, nclasses)#fill me 

    def forward(self, src, src_mask):
        # base model
        x = self.base.forward(src,src_mask) #fill me
        # classifier model
        output = self.classifier.forward(x) #fill me
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, nhid, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, nhid)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, nhid, 2).float() * (-math.log(10000.0) / nhid)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

s = spm.SentencePieceProcessor(model_file='sentencepiece.french.model') #load sentencepiece model

letter_counts = {}
for i in sequences_train :
  protein_seq = i
  for letter in protein_seq:
      letter =s.encode_as_pieces(letter)[0]
      if letter in letter_counts:
          letter_counts[letter] += 1
      else:
          letter_counts[letter] = 1

print(letter_counts)

letter_counts1 = {}
for i in sequences_test :
  protein_seq = i
  for letter in protein_seq:
      letter =s.encode_as_pieces(letter)[0]
      if letter in letter_counts1:
          letter_counts1[letter] += 1
      else:
          letter_counts1[letter] = 1

print(letter_counts1)

path_vocab = letter_counts.keys()
token2ind = {"<sos>": 0, "<pad>": 1, "<eos>": 2, "<oov>": 3} # the 4 first indices are reserved to special tokens
for idx, line in enumerate(path_vocab):
    word = line.split()[0].strip()
    token2ind[word] = idx + 4 #fill me

ind2token = {v: k for k, v in token2ind.items()} #fill me



class Dataset(Dataset):
    def __init__(
        self,
        path_documents,
        path_labels=None,
        token2ind={},
        max_len=512,
        task="classification",
    ):
        self.task = task
        self.max_len = max_len
        self.token2ind = token2ind
        self.documents = []
        self.labels = []
        with open(path_documents, "r", encoding="utf-8") as f1:
            for line in f1:
                self.documents.append(line.strip().strip('\n'))
        if task == "classification":
            with open(path_labels, "r", encoding="utf-8") as f1:
                for line in f1:
                    self.labels.append(int(line.strip().strip('\n')))
            assert len(self.labels) == len(self.documents)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        sequence = self.documents[index].split()
        if len(sequence) > self.max_len - 1:
            sequence = sequence[: self.max_len - 1]
        #source_sequence = [self.token2ind["<sos>"]] + [self.token2ind[str(i)] for i in sequence if i in token2ind.keys()] #fill me (constract the input sequence using token2ind, sequence and special tokens)
        source_sequence = [self.token2ind['<sos>']] + [
                           self.token2ind[w] 
                           if w in self.token2ind 
                           else self.token2ind['<oov>']
                           for w in sequence
                           ] 
     
        # if self.task == "language_modeling":
        #     target = source_sequence[1:]
        #     target.append(self.token2ind["<eos>"])
        # elif self.task == "classification":
        #     target = [self.labels[index]]
        target = [self.labels[index]]
        sample = {
            "source_sequence": torch.tensor(source_sequence),
            "target": torch.tensor(target),
        }
        return sample


def MyCollator(batch):
    source_sequences = pad_sequence(
        #we use padding to match the length of the sequences in the same batch
        [sample["source_sequence"] for sample in batch], padding_value=token2ind["<pad>"]
    )
    target = pad_sequence(
        [sample["target"] for sample in batch], padding_value=token2ind["<pad>"]
    )
    return source_sequences, target.reshape(-1)


def get_loader(
    path_documents,
    path_labels=None,
    token2ind={},
    max_len=512,
    batch_size=8,
    task="classification",
):
    dataset = Dataset(
        path_documents,
        path_labels=path_labels,
        token2ind=token2ind,
        max_len=512,
        task=task,
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MyCollator,
        pin_memory=True,
        drop_last=True,
    )
    return data_loader

def train(
    path_data_train,
    path_labels_train=None,
    path_data_valid=None,
    save_interval=-1,
    log_interval=5,
    task="classification",
    batch_size=16,
):
    model.train()
    total_loss = 0.0
    ntokens = len(token2ind)
    data_loader = get_loader(
        path_data_train,
        path_labels_train,
        token2ind,
        task=task,
        batch_size=batch_size,
    )
    
    losses = []
    for idx, data in enumerate(data_loader): #step 1
        optimizer.zero_grad()
        src_mask = model.base.generate_square_subsequent_mask(data[0].size(0)).to(
            device
        )
        input = data[0].to(device)
        output = model(input, src_mask) #step 2
        if task == 'classification':
            #last vector only
            output = output[-1]#fill me 
        output = output.view(-1, output.shape[-1])
        target = data[1].long() #fill me
        target = target.to(device)
        # print( output,target)
        loss = torch.nn.CrossEntropyLoss()(output,target) #fill me, Cross entropy check next cells
        #fill me step 3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # prevent exploding gradient 
        #fill me step 4
        optimizer.step()
        total_loss += loss.item() 
        if idx % log_interval == 0 and idx > 0:
            cur_loss = total_loss / log_interval
            print(
                "| epoch {:3d} | {:5d}/{:5d} steps | "
                "loss {:5.5f} | ppl {:8.3f}".format(
                    epoch, idx, len(data_loader), cur_loss, math.exp(cur_loss),
                )
            )
            losses.append(cur_loss)
            total_loss = 0
    return losses

# a function to evaluate the validation accuracy of the model.
def evaluate_accuracy(data_loader):
    #to be implemented
    total_correct_predictions = total_length = 0
    for idx, data in enumerate(data_loader): #step 1
        model.eval()
        src_mask = model.base.generate_square_subsequent_mask(data[0].size(0)).to(
            device
        )
        input = data[0].to(device)

        output = model(input, src_mask) #step 2
        output = output[-1]
        output = output.view(-1, output.shape[-1])
        max_prob_indice = output.argmax(-1) 

        target =  data[1] 
        target = target.to(device)

        correct_predictions = torch.sum(target == max_prob_indice)
        total_length += target.shape[0]
        total_correct_predictions += correct_predictions.item()

    return total_correct_predictions/total_length

sequences_train_ , sequences_val, y_train_, y_val = train_test_split(sequences_train,y_train ,
                                   random_state=104, 
                                   test_size=0.1, 
                                   shuffle=True)
  
f= open("data/train.sequences.spm","w+", encoding="utf-8")

for i in sequences_train_:
    seq = " ".join(s.encode(
        " ".join(
            list(i)
            ),out_type=str))
    f.write(seq+' \n')
f.close()

f= open("data/test.sequences.spm","w+", encoding="utf-8")

for i in sequences_val:
    seq = " ".join(s.encode(
        " ".join(
            list(i)
            ),out_type=str))
    f.write(seq+' \n')
f.close()

f= open("data/train.label","w+", encoding="utf-8")

for i in y_train_:

    f.write(f'{i}'+' \n')
f.close()

f= open("data/test.label","w+", encoding="utf-8")

for i in y_val:
    f.write(f'{i}'+' \n')
f.close()


path_data_train = "data/train.sequences.spm"
path_labels_train = "data/train.label"

path_data_valid = "data/test.sequences.spm"
path_labels_valid = "data/test.label"


from_scratch_settings = [False]

ntokens = len(token2ind)#fill me # the size of vocabulary
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 8  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value

from_scratch_valid_acc = []
pretrained_valid_acc = []
lr = 0.0001
num_classes = 18  # for classification task only
epochs = 450

model = Model(ntokens, nhead, nhid, nlayers, num_classes, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if not from_scratch_settings[0]:
    print("=====PRETRAINED MODEL======")
    #load checkpoint
    checkpoint = torch.load("Models/Test_light_model_1050.pt")
    #load state dict
    model.base.load_state_dict(checkpoint['model_state_dict'])
else:
    print("=====Trainig FROM SCRATCH======")

for epoch in range(1, epochs + 1):
    train(
        path_data_train,
        path_labels_train,
        save_interval=-1,
        task='classification',
        batch_size=8,
        log_interval=50,
    )
    acc = evaluate_accuracy(
        get_loader(
            path_data_valid,
            path_labels_valid,
            token2ind=token2ind,
            batch_size=8,
            task='classification',
        )
    )
    if epoch%50 ==0 :
        torch.save({"model_state_dict": model.base.state_dict(),}, f"Models/Test_light_model_{epoch+1050}.pt")
    if from_scratch_settings:
        from_scratch_valid_acc.append(acc)
    else:
        pretrained_valid_acc.append(acc)
    print("Accuracy", acc)
print()


import matplotlib.pyplot as plt


plt.plot(pretrained_valid_acc, label="Pretrained accuracy")
plt.plot(from_scratch_valid_acc, label="From scratch accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()