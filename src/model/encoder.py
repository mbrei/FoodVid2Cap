import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    '''
    Encodes ResNet_features of video frames and provides functionality for encoding in Seq_to_Seq Model.
    '''
    def __init__(self, num_features, hidden_size=256, num_layers=1, num_dnn_layers=0, dropout=0, bidirectional=True,
                 device="cuda"):
        '''
        Initialize the encoder module.

        Input:
        :param num_features: integer, number of visual features in each video frame
        :param hidden_size: integer, hidden dimension of encoder
        :param num_layers: integer, number of desired layers of GRU
        :param num_dnn_layers: integer, number of desired linear layers applied before GRU
        :param dropout: double, dropout probability of dropout layer on outputs of each GRU layer except the last layer
        :param bidirectional: integer, 2 if bidirectional is used, else 1
        :param device: string, specifies availability of CUDA
        '''
        super(EncoderRNN, self).__init__()
        self.input_size = num_features
        self.hidden_size = hidden_size
        self.layers = num_layers
        self.dnn_layers = num_dnn_layers
        self.dropout = dropout
        self.bi = bidirectional
        self.device = torch.device(device)

        # Determine bidirectional
        num_dir = 1
        if self.bi:
            num_dir = 2

        # Initialize linear layers applied before GRU
        if self.dnn_layers > 0:
            for i in range(self.dnn_layers):
                self.add_module('dnn_' + str(i), nn.Linear(
                    in_features=self.input_size if i == 0 else self.hidden_size,
                    out_features=self.hidden_size
                ))

        # Compute input dimension of GRU
        gru_input_dim = self.input_size if self.dnn_layers == 0 else self.hidden_size
        self.rnn = nn.GRU(
            gru_input_dim,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True)

        self.fc = nn.Linear(self.hidden_size * num_dir, self.hidden_size)

    def run_dnn(self, x):
        '''
        Executes linear layers before GRU.

        Input:
        :param x: torch.tensor input vector

        Output:
        :return: x: torch.tensor vector after applying all linear layers
        '''

        for i in range(self.dnn_layers):
            x = F.relu(getattr(self, 'dnn_' + str(i))(x))
        return x

    def forward(self, inputs, input_lengths):
        '''
        Performs forward pass of the Encoder.

        Input:
        :param inputs: torch.tensor, input feature vector
        :param input_lengths: torch.tensor, lengths of video frames

        Output:
        :return: torch.tensor, torch.tensor: output vector and hidden state of encoder
        '''

        batch_size = inputs.size(0)

        # Initialize hidden state for GRU
        hidden = self.init_hidden(batch_size).to(self.device)

        # Execute linear layers on input features
        if self.dnn_layers > 0:
            inputs = self.run_dnn(inputs)

        # Create seqeunce of packed input vectors
        x = pack_padded_sequence(inputs, input_lengths, batch_first=True).to(self.device)

        # Apply GRU
        output, hidden = self.rnn(x, hidden)
        output, hidden = output.to(self.device), hidden.to(self.device)

        # Reverse packing, [batch_size, max_seq_len,hidden_size * bi_directional]
        output, _ = pad_packed_sequence(output, batch_first=True, padding_value=0)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return output, hidden

    def init_hidden(self, batch_size):
        '''
        Initialize hidden state for GRU.

        Input:
        :param batch_size: integer, number of batches
        Output
        :return: torch.tensor, filled with zeros in suitable dimensions
        '''
        if self.bi:
            return Variable(torch.zeros((self.layers * 2, batch_size, self.hidden_size)))
        else:
            return Variable(torch.zeros((self.layers, batch_size, self.hidden_size)))