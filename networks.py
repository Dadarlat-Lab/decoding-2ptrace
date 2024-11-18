
import torch
import torch.nn as nn

import os
import random
import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_checkpoint(model, path_ckpt):
    loaded = torch.load(path_ckpt)
    model.load_state_dict(loaded['model_state_dict'])

    return model

def get_masked_pred(pred, target):
    if isinstance(target, torch.Tensor):
        mask = target.bool().type(pred.dtype)
    elif isinstance(target, np.ndarray):
        mask = target!=0
    
    pred_masked = pred*mask

    return pred_masked

def reverse_mask(pred_stack, target_stack):
    '''
    pred_stack, target_stack: numpy array, dim=2
    '''
    if pred_stack.ndim != 2 or target_stack.ndim!=2:
        print('dim!=2')
        exit()

    idx = np.any(target_stack, axis=1)
    target_select = target_stack[idx, :]
    pred_select = pred_stack[idx, :]

    return pred_select, target_select

class loss_seq2seq_mse(nn.Module):
    def __init__(self):
        super(loss_seq2seq_mse, self).__init__()
    
    def forward(self, pred, target):
        pred_masked = get_masked_pred(pred, target)
        mse = torch.mean((pred_masked-target)*(pred_masked-target)) #Checked it is same to torch.nn.MSELoss()

        return mse
        
class simpleLSTM_many2many(nn.Module):
    '''
    https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb
    https://wandb.ai/sauravmaheshkar/LSTM-PyTorch/reports/Using-LSTM-in-PyTorch-A-Tutorial-With-Examples--VmlldzoxMDA2NTA5
    '''
    def __init__(self, input_size, hidden_size, num_layers, num_output, bi=False, dropout=0):
        super(simpleLSTM_many2many, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_output = num_output
        self.dropout = dropout

        self.lstm = None
        if dropout != 0 and num_layers == 1:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bi)
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bi)

        if bi:
            self.fc = nn.Linear(hidden_size*2, num_output)
        else:
            self.fc = nn.Linear(hidden_size, num_output)

    def forward(self, x):
        flag_print = False

        if self.dropout != 0 and self.num_layers == 1:
            output_lstm, _ = self.lstm(x)
            output = self.dropout_layer(output_lstm)
        else:
            output, (final_hidden_state, final_cell_state) = self.lstm(x)
        if flag_print: 
            print('lstm: ', output.shape) #(batch, seq, hidden_size)
            # print('final_hidden_state: ', final_hidden_state.shape)
            # print('final_cell_state: ', final_cell_state.shape)
            # print('final_hidden_state[-1]: ', final_hidden_state[-1].shape)

        fc = self.fc(output)
        if flag_print: print('fc: ', fc.shape) #(batch, seq, num_output)
        
        return fc

class rnn_encdec(nn.Module):
    '''
    encoder decoder same sequence length
    '''
    def __init__(self, input_size, hidden_size, num_layers, output_size, device, dropout=None, bi=False, layer_type='lstm', flag_return_enc_output=False):
        super(rnn_encdec, self).__init__()

        self.device = device
        self.layer_type = layer_type
        self.hidden_size = hidden_size
        self.flag_return_enc_output = flag_return_enc_output

        self.output_size = output_size

        if dropout is None:
            if layer_type=='lstm':
                self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bi)
                self.decoder = nn.LSTM(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bi)
            elif layer_type=='gru':
                self.encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bi)
                self.decoder = nn.GRU(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bi)
            elif layer_type=='rnn':
                self.encoder = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bi)
                self.decoder = nn.RNN(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bi)

            if bi is False:
                self.fc = nn.Linear(hidden_size, output_size)
            else:
                self.fc = nn.Linear(hidden_size*2, output_size)
        #else: 
          
    def forward(self, x): 
        flag_print = False

        if flag_print:
            print('x: ', x.shape)  
            # x:  torch.Size([84, 5, 3361]) #(batch, seq_neural, neurons)

        # transpose -> batch_first=False    
        x = torch.transpose(x, 1, 0)
        if flag_print:
            print('x transpose: ', x.shape)
            # x transpose:  torch.Size([5, 84, 3361]) #(seq_neural, batch, neurons), pytorch input: (seq, batch, input_size)

        #tensor to store decoder outputs
        batch_size = x.shape[1]
        seq_neural = x.shape[0]
        outputs = torch.zeros(seq_neural, batch_size, self.output_size).to(self.device)
        if flag_print:
            print('outputs: ', outputs.shape)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        if self.layer_type == 'lstm': 
            output_enc, (hidden, cell) = self.encoder(x)
            if flag_print: 
                print('output_enc: ', output_enc.shape) #torch.Size([5, 160, 795]) = (seq_neuron, batch,)
                print('hidden, enc: ', hidden.shape) 
                print('cell, enc: ', cell.shape) 
                #print(torch.equal(output_enc[-1,:,:], hidden[-1,:,:]))
                # hidden_enc:  torch.Size([3, 84, 128]) #(num_layers, batch, hidden_size)
                # cell_enc:  torch.Size([3, 84, 128]) #(num_layers, batch, hidden_size)
        elif self.layer_type == 'gru' or self.layer_type == 'rnn':
            _, hidden = self.encoder(x)
            if flag_print: 
                print('hidden, enc: ', hidden.shape) 
        hidden_enc = hidden

        #first input to the decoder is the  tokens
        input_dec = torch.zeros((batch_size, self.output_size)).to(self.device)
        if flag_print:
            print('input_dec: ', input_dec.shape)
            # input_dec:  torch.Size([84, 8]) #(batch, output_size)

        for t in range(seq_neural):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            input_dec = input_dec.unsqueeze(0) 
            if flag_print:
                print(t, '/', str(seq_neural))
                print('input_dec: ', input_dec.shape)
                # input_dec:  torch.Size([1, 84, 8]) #(1, batch, output_size)

            if self.layer_type == 'lstm':
                output, (hidden, cell) = self.decoder(input_dec, (hidden, cell))
                if flag_print:
                    print('output: ', output.shape)
                    print('hidden: ', hidden.shape)
                    print('cell: ', cell.shape)
                    # output:  torch.Size([1, 84, 128])
                    # hidden:  torch.Size([3, 84, 128])
                    # cell:  torch.Size([3, 84, 128])
            elif self.layer_type == 'gru' or self.layer_type == 'rnn':
                output, hidden = self.decoder(input_dec, hidden)
                if flag_print:
                    print('output: ', output.shape)
                    print('hidden: ', hidden.shape)
            
            if torch.isnan(output).any():
                print("There is nan in output before fc.")
                #exit()
                return None
            if output.shape[-1] !=  self.hidden_size:
                print("Shape is wrong. output before fc")
                print('output: ', output.shape)
                exit()
            #print('output before fc: ', output.shape)

            output = self.fc(output)
            if flag_print:
                print('fc output: ', output.shape)
                # fc output:  torch.Size([1, 84, 8])
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            # #decide if we are going to use teacher forcing or not
            # #teacher_forcing_ratio = 0.5
            # teacher_force = random.random() < teacher_force_ratio
            # if flag_print:
            #     print('teacher_force: ', teacher_force)
            # # if teacher_force:
            # #     print("teacher_force is True. Something is wrong.")
            # #     exit()
            # # if teacher_force:
            # #     print('teacher_force: ', teacher_force)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            # input_dec = trg[t] if teacher_force else output[0]
            input_dec = output[0]
            if flag_print:
                print('input_dec: ', input_dec.shape)
                # input_dec:  torch.Size([84, 8])

        if self.flag_return_enc_output is False:
            return torch.transpose(outputs, 1, 0)
        else:
            return torch.transpose(outputs, 1, 0), hidden_enc, output_enc
        
class seq2seq(nn.Module):
    '''
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
    https://stackoverflow.com/questions/49283435/how-does-batching-work-in-a-seq2seq-model-in-pytorch
    https://d2l.ai/chapter_recurrent-modern/seq2seq.html
    '''
    def __init__(self, input_size, hidden_size, num_layers, output_size, device, dropout=None, bi=False, layer_type='lstm', flag_return_enc_output=False):
        super(seq2seq, self).__init__()

        self.device = device
        self.layer_type = layer_type
        self.hidden_size = hidden_size
        self.flag_return_enc_output = flag_return_enc_output

        if dropout is None:
            if layer_type=='lstm':
                self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bi)
                self.decoder = nn.LSTM(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bi)
            elif layer_type=='gru':
                self.encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bi)
                self.decoder = nn.GRU(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bi)
            elif layer_type=='rnn':
                self.encoder = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bi)
                self.decoder = nn.RNN(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bi)

            if bi is False:
                self.fc = nn.Linear(hidden_size, output_size)
            else:
                self.fc = nn.Linear(hidden_size*2, output_size)
        #else: 
          
    def forward(self, x, trg): #, teacher_force_ratio):
        flag_print = False

        if flag_print:
            print('x: ', x.shape)  
            print('trg: ', trg.shape) 
            # x:  torch.Size([84, 5, 3361]) #(batch, seq_neural, neurons)
            # trg:  torch.Size([84, 20, 8]) #(batch, seq_coord, limb)

        # transpose -> batch_first=False    
        x = torch.transpose(x, 1, 0)
        trg = torch.transpose(trg, 1, 0)
        if flag_print:
            print('x transpose: ', x.shape)
            print('trg transpose: ', trg.shape) 
            # x transpose:  torch.Size([5, 84, 3361]) #(seq_neural, batch, neurons), pytorch input: (seq, batch, input_size)
            # trg transpose:  torch.Size([20, 84, 8]) #(seq_coord, batch, limb)

        #tensor to store decoder outputs
        batch_size = x.shape[1]
        seq_coord = trg.shape[0]
        output_size = trg.shape[2]
        outputs = torch.zeros(seq_coord, batch_size, output_size).to(self.device)
        if flag_print:
            print('outputs: ', outputs.shape)
            # outputs:  torch.Size([20, 84, 8]) #(seq_coord, batch, output_size)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        if self.layer_type == 'lstm': 
            output_enc, (hidden, cell) = self.encoder(x)
            if flag_print: 
                print('output_enc: ', output_enc.shape) #torch.Size([5, 160, 795]) = (seq_neuron, batch,)
                print('hidden, enc: ', hidden.shape) 
                print('cell, enc: ', cell.shape) 
                #print(torch.equal(output_enc[-1,:,:], hidden[-1,:,:]))
                # hidden_enc:  torch.Size([3, 84, 128]) #(num_layers, batch, hidden_size)
                # cell_enc:  torch.Size([3, 84, 128]) #(num_layers, batch, hidden_size)
        elif self.layer_type == 'gru' or self.layer_type == 'rnn':
            _, hidden = self.encoder(x)
            if flag_print: 
                print('hidden, enc: ', hidden.shape) 
        hidden_enc = hidden

        #first input to the decoder is the  tokens
        input_dec = torch.zeros(trg[0,:].shape).to(self.device)
        if flag_print:
            print('input_dec: ', input_dec.shape)
            # input_dec:  torch.Size([84, 8]) #(batch, output_size)

        for t in range(seq_coord):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            input_dec = input_dec.unsqueeze(0) 
            if flag_print:
                print(t, '/', str(seq_coord))
                print('input_dec: ', input_dec.shape)
                # input_dec:  torch.Size([1, 84, 8]) #(1, batch, output_size)

            if self.layer_type == 'lstm':
                output, (hidden, cell) = self.decoder(input_dec, (hidden, cell))
                if flag_print:
                    print('output: ', output.shape)
                    print('hidden: ', hidden.shape)
                    print('cell: ', cell.shape)
                    # output:  torch.Size([1, 84, 128])
                    # hidden:  torch.Size([3, 84, 128])
                    # cell:  torch.Size([3, 84, 128])
            elif self.layer_type == 'gru' or self.layer_type == 'rnn':
                output, hidden = self.decoder(input_dec, hidden)
                if flag_print:
                    print('output: ', output.shape)
                    print('hidden: ', hidden.shape)
            
            if torch.isnan(output).any():
                print("There is nan in output before fc.")
                #exit()
                return None
            if output.shape[-1] !=  self.hidden_size:
                print("Shape is wrong. output before fc")
                print('output: ', output.shape)
                exit()
            #print('output before fc: ', output.shape)

            output = self.fc(output)
            if flag_print:
                print('fc output: ', output.shape)
                # fc output:  torch.Size([1, 84, 8])
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            # #decide if we are going to use teacher forcing or not
            # #teacher_forcing_ratio = 0.5
            # teacher_force = random.random() < teacher_force_ratio
            # if flag_print:
            #     print('teacher_force: ', teacher_force)
            # # if teacher_force:
            # #     print("teacher_force is True. Something is wrong.")
            # #     exit()
            # # if teacher_force:
            # #     print('teacher_force: ', teacher_force)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            # input_dec = trg[t] if teacher_force else output[0]
            input_dec = output[0]
            if flag_print:
                print('input_dec: ', input_dec.shape)
                # input_dec:  torch.Size([84, 8])

        if self.flag_return_enc_output is False:
            return torch.transpose(outputs, 1, 0)
        else:
            return torch.transpose(outputs, 1, 0), hidden_enc, output_enc



        
