import pandas as pd
import pickle
from torchtext import data 
from model import classifier
from dataset import PandasDataFrame
import torch

print('lstm_unique_model_4236')
size_of_vocab = 18344
embedding_dim = 40
num_hidden_nodes = 128
num_output_nodes = 96
num_layers = 2
bidirection = True
dropout = 0.2


model =  classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                   bidirectional = bidirection, dropout = dropout)

model.load_state_dict(torch.load('./saved_weights.pt', map_location=torch.device('cpu')))
model.eval()

# 'data/task1_test_for_user.parquet' 'processed_train.csv'

test = pd.read_parquet('data/task1_test_for_user.parquet')

test['item_name'] = test['item_name'].apply(lambda x: 'a' if x == '' else x)

import dill 

TEXT = data.Field(batch_first=False, include_lengths=True, pad_token='<pad>', unk_token='<unk>')

with open("./LABEL.FIELD", "rb")as f:
     LABEL=dill.load(f)

test_ds = PandasDataFrame(
  text_field=TEXT, label_field=None, df = test, is_test=True)

# print(test_ds)
device = torch.device('cpu')  

model = model.to(device)
BATCH_SIZE = 64


#Load an iterator
valid_iterator = data.Iterator(
    test_ds, 
    train=False,
    batch_size = BATCH_SIZE,
    sort_within_batch=False,
    device = device)

model.enforce_sorted = False
def predict(model, iterator):

    #deactivating dropout layers
    model.eval()
    preds = []
    #deactivates autograd
    with torch.no_grad():
    
        for batch in iterator:
        
            #retrieve text and no. of words
            text, text_lengths = batch.text
            
            #convert to 1d tensor
            preds += [LABEL.vocab.itos[n] for n in model(text, text_lengths.cpu()).argmax(axis=1)]

            #preds += [71 for i in range(len(text))]
        
    return preds

#predict(model, valid_iterator)

preds = predict(model, valid_iterator)# [71 for i in range(test.shape[0])] #predict(model, valid_iterator)

res = pd.DataFrame(preds, columns=['pred'])
res['id'] = test['id']

res[['id', 'pred']].to_csv('answers.csv', index=None)
