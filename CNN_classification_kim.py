
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torchtext
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


data = torchtext.data


# In[3]:


class_label = data.Field(use_vocab=True, sequential=False, 
                         lower=True, is_target=True)
sentence = data.Field(use_vocab=True, sequential=True,  
                     lower=True, fix_length=105,
                     init_token = "<s>", eos_token = "<\s>",)
train, validation = data.TabularDataset.splits(path="topicclass/",
                            train='topicclass_train.tsv',
                            validation='topicclass_valid.tsv',
                            skip_header=False, format='tsv',
                            fields = [
                                ('class_label', class_label), 
                                ('sentence', sentence)
                            ]
                           )
test = data.TabularDataset(path="topicclass/topicclass_test.tsv",
                          skip_header=True, format='tsv',
                          fields = [
                              ('class_label', None),
                              ('sentence', sentence)
                          ])


# In[4]:


class_label.build_vocab(train, validation)
sentence.build_vocab(train, validation, test, min_freq = 5, vectors = 'glove.840B.300d')


# In[21]:


params = {
        "batch_size": 4096,
        "embedding_size": 300,
        "sentence_length": 105,
        "vocab_size": len(sentence.vocab),
        "n_classes": len(class_label.vocab),
        "filter_sizes": [3, 5, 7],
        "filter_num": [100, 100, 100],
        "PreTrained" : sentence.vocab.vectors
}


# In[22]:


class CNN(nn.Module):
    def __init__(self, **params):
        super(CNN, self).__init__()
        
        
        self.batch_size = params["batch_size"]
        self.sentence_length = params["sentence_length"]
        self.embedding_size = params["embedding_size"]
        self.vocab_size = params["vocab_size"]
        self.n_classes = params["n_classes"]
        self.filter_sizes = params["filter_sizes"]
        self.filter_num = params["filter_num"]
        self.PreTrained = params["PreTrained"]
        self.in_channel = 1
  
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(self.PreTrained)

        
        self.conv0 = nn.Conv1d(self.in_channel, self.filter_num[0], self.embedding_size * self.filter_sizes[0], stride=self.embedding_size)
        self.conv1 = nn.Conv1d(self.in_channel, self.filter_num[1], self.embedding_size * self.filter_sizes[1], stride=self.embedding_size)
        self.conv2 = nn.Conv1d(self.in_channel, self.filter_num[2], self.embedding_size * self.filter_sizes[2], stride=self.embedding_size)
        
        self.linear1 = nn.Linear(sum(self.filter_num), sum(self.filter_num))
        self.linear2 = nn.Linear(sum(self.filter_num), self.n_classes)
        
    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.embedding_size * self.sentence_length


        
        x1 = F.max_pool1d(F.relu(self.conv0(x)), self.sentence_length - self.filter_sizes[0] + 1).view(-1, self.filter_num[0])
                      
        x2 = F.max_pool1d(F.relu(self.conv1(x)), self.sentence_length - self.filter_sizes[1] + 1).view(-1, self.filter_num[1])
                      
        x3 = F.max_pool1d(F.relu(self.conv2(x)), self.sentence_length - self.filter_sizes[2] + 1).view(-1, self.filter_num[2]) 

        
        

        x = torch.cat([x1,x2,x3], 1)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.linear2(x))
        
        return x


# In[41]:


model = CNN(**params) 


# In[12]:


type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()


# In[13]:


criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr = 8e-4, weight_decay=8e-4)


# In[14]:


train_iter = data.BucketIterator(
 (train), 
 batch_size=4096,
 device=torch.device('cuda'),
 shuffle = True,
 repeat=False 
)

valid_iter = data.BucketIterator(
 (validation), 
 batch_size=4096,
 device=torch.device('cuda'),
 shuffle = False,
 repeat=False 
)


# In[16]:


for epoch in range(1,21):
    train_loss = 0
    for batch in train_iter: 
        model.train()
        opt.zero_grad()
        text, targets = batch.sentence, batch.class_label
        text = torch.t(text)
        prediction = model(text)
        loss = criterion(prediction.view(-1,len(class_label.vocab)),targets.view(-1))
        loss.backward()
        opt.step()
 
        train_loss += loss.data.item() * text.size(0)
        model.eval()
         
    train_loss /= len(train.examples)
    
    with torch.no_grad():
        val_loss = 0.0
        model.eval() 
        for batch in valid_iter:
            text, targets = batch.sentence, batch.class_label
            text = torch.t(text)
            prediction = model(text)
            loss = criterion(prediction,targets)
            val_loss += loss.data.item() * text.size(0)
            val_acc = float(100*(prediction.argmax(dim = 1) == targets).sum())/len(validation.examples)
        val_loss /= len(validation.examples)
    if ((train_loss < 0.8) & (val_loss < 0.73) & ( val_acc > 82.5)):
        torch.save(model, "model_{}".format(epoch))
        
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation acc: {:.4f}'.format(epoch, train_loss, val_loss, val_acc))


# In[25]:


model = torch.load("model_15")

with torch.no_grad():
    val_loss =0.0
    model.eval()
    for batch in valid_iter:
        text, targets = batch.sentence, batch.class_label
        text = torch.t(text)
        prediction = model(text)
        loss = criterion(prediction,targets)
        val_loss += loss.data.item() * text.size(0)
        val_acc = float(100*(prediction.argmax(dim = 1) == targets).sum())/len(validation.examples)
    val_loss /= len(validation.examples)
    print('Validation Loss: {:.4f}, Validation acc: {:.4f}'.format(val_loss, val_acc))


# In[26]:


test_iter = data.BucketIterator(
 (test), # we pass in the datasets we want the iterator to draw data from
 batch_size=4096,
 device=torch.device('cuda'),
 shuffle = False,# if you want to use the GPU, specify the GPU number here
 repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
)


# In[27]:


val_label =np.array([None]*643)
test_label =np.array([None]*696)


# In[28]:


val_label[np.array(prediction.argmax(dim =1)) == 1] = 'music '
val_label[np.array(prediction.argmax(dim =1)) == 2] = 'sports and recreation '
val_label[np.array(prediction.argmax(dim =1)) == 3] = 'natural sciences '
val_label[np.array(prediction.argmax(dim =1)) == 4] = 'warfare '
val_label[np.array(prediction.argmax(dim =1)) == 5] = 'media and drama '
val_label[np.array(prediction.argmax(dim =1)) == 6] = 'social sciences and society '
val_label[np.array(prediction.argmax(dim =1)) == 7] = 'history '
val_label[np.array(prediction.argmax(dim =1)) == 8] = 'engineering and technology '
val_label[np.array(prediction.argmax(dim =1)) == 9] = 'geography and places '
val_label[np.array(prediction.argmax(dim =1)) == 10] = 'video games '
val_label[np.array(prediction.argmax(dim =1)) == 11] = 'art and architecture '
val_label[np.array(prediction.argmax(dim =1)) == 12] = 'language and literature '
val_label[np.array(prediction.argmax(dim =1)) == 13] = 'philosophy and religion '
val_label[np.array(prediction.argmax(dim =1)) == 14] = 'agriculture, food and drink '
val_label[np.array(prediction.argmax(dim =1)) == 15] = 'miscellaneous '
val_label[np.array(prediction.argmax(dim =1)) == 16] = 'mathematics '


# In[29]:


with torch.no_grad():
    model.eval()
    for batch in test_iter:
        text = batch.sentence
        text = torch.t(text)
        prediction = model(text)


# In[30]:


test_label[np.array(prediction.argmax(dim =1)) == 1] = 'music '
test_label[np.array(prediction.argmax(dim =1)) == 2] = 'sports and recreation '
test_label[np.array(prediction.argmax(dim =1)) == 3] = 'natural sciences '
test_label[np.array(prediction.argmax(dim =1)) == 4] = 'warfare '
test_label[np.array(prediction.argmax(dim =1)) == 5] = 'media and drama '
test_label[np.array(prediction.argmax(dim =1)) == 6] = 'social sciences and society '
test_label[np.array(prediction.argmax(dim =1)) == 7] = 'history '
test_label[np.array(prediction.argmax(dim =1)) == 8] = 'engineering and technology '
test_label[np.array(prediction.argmax(dim =1)) == 9] = 'geography and places '
test_label[np.array(prediction.argmax(dim =1)) == 10] = 'video games '
test_label[np.array(prediction.argmax(dim =1)) == 11] = 'art and architecture '
test_label[np.array(prediction.argmax(dim =1)) == 12] = 'language and literature '
test_label[np.array(prediction.argmax(dim =1)) == 13] = 'philosophy and religion '
test_label[np.array(prediction.argmax(dim =1)) == 14] = 'agriculture, food and drink '
test_label[np.array(prediction.argmax(dim =1)) == 15] = 'miscellaneous '
test_label[np.array(prediction.argmax(dim =1)) == 16] = 'mathematics '


# In[36]:


np.savetxt("validation_labels", val_label, newline="\n", fmt = "%s")
np.savetxt("test_labels", test_label, newline="\n", fmt = "%s")

