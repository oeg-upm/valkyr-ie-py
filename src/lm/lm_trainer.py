#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:39:51 2021

@author: Pablo
"""

#!pip install transformers


from transformers import RobertaTokenizer
from tqdm.auto import tqdm
import logging
import torch


def readlines(file):
  with open(file, 'r', encoding='utf-8') as fp:
    lines = fp.read().split('\n')
    fp.close()
    return lines 


model_name='d4c_lm'
max_files=300
path='/home/s730/PROJECT/data/'

logging.basicConfig(filename='lm_trainer.log', level=logging.INFO)

# initialize the tokenizer using the tokenizer we initialized and saved to file
tokenizer = RobertaTokenizer.from_pretrained(model_name, max_len=512)


'''
# test our tokenizer on a simple sentence
tokens = tokenizer('hello word')

tokenizer('hello word')

print(tokens)
'''


"""## Pipeline"""


logging.info('Starting to read files')
print('Starting to read files')


labels_arr=[]
mask_arr=[]
for a in range(0,max_files):
    print(a)
    logging.info(a)
    lines= readlines(path+'text_'+str(a)+'.txt')
    batch = tokenizer(text=lines, max_length=512, padding='max_length', truncation=True)
    labels_arr.extend(batch['input_ids'])
    mask_arr.extend(batch['attention_mask'])

logging.info('Finish')
print('Finished reading')
labels = torch.tensor(labels_arr)
mask = torch.tensor(mask_arr)

'''
for i in range(0,max_files):
    logging.info('Iteration: '+str(i))
    with open(path+'text_'+i+'.txt', 'r', encoding='utf-8') as fp:
        lines.extend(fp.read().split('\n'))


logging.info('End to read files')
logging.info('Tokenization')

batch = tokenizer(text=lines, max_length=512, padding='max_length', truncation=True)




logging.info('Preparing')

labels = torch.tensor(batch['input_ids'])
mask = torch.tensor(batch['attention_mask'])

'''




#labels = torch.tensor([x.input_ids for x in batch])
#mask = torch.tensor([x.attention_mask for x in batch])

# make copy of labels tensor, this will be input_ids
input_ids = labels.detach().clone()
# create random array of floats with equal dims to input_ids
rand = torch.rand(input_ids.shape)
# mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
# loop through each row in input_ids tensor (cannot do in parallel)
for i in range(input_ids.shape[0]):
    # get indices of mask positions from mask array
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    # mask input_ids
    input_ids[i, selection] = 3  # our custom [MASK] token == 3

#input_ids.shape

#input_ids[0][:200]

encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}



logging.info('Creating dataset')
print('Creating dataset')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}

dataset = Dataset(encodings)

loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)



"""## Training"""
logging.info('Model creation')
print('Model creation')
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=30_522,  # we align this to the tokenizer vocab_size
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config)

#torch.cuda.is_available()

logging.info('GPU: '+str(torch.cuda.is_available()))
print('GPU: '+str(torch.cuda.is_available()))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)

from transformers import AdamW

# activate training mode
model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=1e-4)

epochs = 5


logging.info('TRAINING')
print('TRAINING')

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    logging.info(epoch)
    print(epoch)
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())


logging.info('Saved')
model.save_pretrained('./'+model_name)  # and don't forget to save filiBERTo!

"""## Test"""

'''

from transformers import pipeline

fill = pipeline('fill-mask', model='oegberto', tokenizer='oegberto')

fill(f'for the current {fill.tokenizer.mask_token} we will')


'''


