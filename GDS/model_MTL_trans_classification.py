""" Imports """
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class TMTL(nn.Module):
    def __init__(self, dim_input, dim_output, task_num=3, dim_emb=64, dropout_rate=0.2, batch_first=True, device='cuda'):
        super(TMTL, self).__init__()
        self.dropout_rate = dropout_rate
        self.batch_first = batch_first
        self.embedding = nn.Sequential(
            nn.Linear(dim_input, dim_emb, bias=False),
        )
        #init.uniform_(self.embedding[0].weight, a=-1, b=1)
        self.dropout = nn.Dropout(dropout_rate)
        #self.rnn_encoder = nn.GRU(input_size=dim_emb, hidden_size=dim_emb, num_layers=1, batch_first=self.batch_first)
        self.rnn_encoder = nn.RNN(input_size=dim_emb, hidden_size=dim_emb, num_layers=1, batch_first=self.batch_first)

        self.rnn_fc = nn.Linear(in_features=dim_emb, out_features=dim_emb)

        self.tasks_emb = torch.rand(task_num, dim_emb, requires_grad = True).to(device)
        self.key = self.transformer_param(dim_emb)
        self.query = self.transformer_param(dim_emb)
        self.value = self.transformer_param(dim_emb)
        self.softmax_lay = nn.Softmax(dim=1)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=dim_emb, nhead=1, dim_feedforward=dim_emb, dropout= 1., batch_first=True)
        # self.tasks_assign = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.task_model = []
        for i in range(task_num):
            task_out_layer = self.task_prediction(dropout_rate, dim_emb, dim_output)
            self.task_model.append(task_out_layer.to(device))

        self.weight_init()


        #self.classifier = nn.Linear()

    def transformer_param(self, dim_emb):
        output = nn.Sequential(
            nn.Linear(in_features=dim_emb, out_features=dim_emb),
            nn.Dropout(0.6)
        )
        init.uniform_(output[0].weight, a=-1, b=1)
        init.zeros_(output[0].bias)
        return output

    def weight_init(self):
        # init.xavier_uniform_(self.embedding[1].weight) # initialize linear layer, other choice: xavier_normal
        init.uniform_(self.embedding[0].weight, a=-0.5, b=0.5)

        # init.xavier_uniform_(self.alpha_fc.weight)
        init.uniform_(self.rnn_fc.weight, a=-0.5, b=0.5)
        init.zeros_(self.rnn_fc.bias)

    def task_prediction(self, dropout, dim_emb, dim_output):
        output = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=dim_emb, out_features=dim_output),
            # nn.ReLU(),
            #nn.Dropout(p=dropout)
            # nn.Linear(in_features=int(dim_emb/2), out_features=dim_output)
            )
        init.uniform_(output[1].weight, a=-1, b=1)
        init.zeros_(output[1].bias)
        # init.uniform_(output[4].weight, a=-1, b=1)
        # init.zeros_(output[4].bias)
        return output

    def forward(self, x, lengths):
        emb = self.embedding(x)

        packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)
        _, h_n = self.rnn_encoder(packed_input)
        h_n = torch.squeeze(h_n)
        h_n = self.rnn_fc(h_n)

        key = self.key(h_n)
        #value = self.value(h_n)
        value = h_n
        query = self.query(self.tasks_emb)

        att = self.softmax_lay(torch.matmul(key, torch.transpose(query, 0, 1))/np.sqrt(h_n.size()[1]))
        # task_input = torch.matmul(torch.unsqueeze(att, 2), torch.unsqueeze(value, 1))
        # task_input = torch.transpose(task_input, 0, 1)

        output_by_task = []
        for i in range(self.tasks_emb.size()[0]):
            # task_out = self.task_model[i](task_input[i])
            task_out = self.task_model[i](value)
            output_by_task.append(task_out)

        temp1 = torch.stack(output_by_task, dim=1)
        overall_out = torch.matmul(torch.unsqueeze(att, 1), temp1)
        # overall_out = torch.sum(temp1, dim=1)
        out = torch.squeeze(overall_out, 1)
        return out