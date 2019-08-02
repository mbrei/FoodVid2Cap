import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    '''
    Applies attention mechanism on the hidden decoder state and encoder output.
    '''
    def __init__(self, encoder_hid_dem, decoder_hid_dem, bidirectional):
        '''
        Initialize attention module.

        Inputs:
        - encoder_hid_dem: hidden dimension of encoder
        - decoder_hid_dem: hidden dimension of decoder
        - bidirectional: True if bidirectional GRU is used, else False
        '''
        super(Attention, self).__init__()
        self.enc_hid_dim = encoder_hid_dem
        self.dec_hid_dim = decoder_hid_dem
        self.encoder_n_direction = 1;
        if (bidirectional == True):
            self.encoder_n_direction = 2;

        self.attn = nn.Linear((encoder_hid_dem * self.encoder_n_direction) + decoder_hid_dem, decoder_hid_dem)
        self.v = nn.Parameter(torch.rand(decoder_hid_dem))

    def forward(self, hidden, encoder_outputs):
        """
        Computes context vector of attention mechanism.

        Inputs:
        :param hidden: torch.tensor, hidden state of the decoder
        :param encoder_outputs: torch.tensor, output of the decoder

        Output:
        :return: torch.tensor, context vector of attention
        """

        # hidden = [batch_size, dec_hid_dim]
        # encoder_outputs = [batch_size, seq_len, enc_hid_dim * encoder_n_direction]

        # batch_size = encoder_outputs.shape[0]
        # src_len = encoder_outputs.shape[1]

        # hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # hidden          = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * encoder_n_direction]

        # energy = [batch_size, seq_len, dec_hid_dim]
        # energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, dec_hid_dim, seq_len]
        # energy = energy.permute(0, 2, 1)


        # v = [dec_hid_dim] -> [batch_size, 1, dec_hid_dim]
        # v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # attention= [batch, seq_len]
        # attention = torch.bmm(v, energy).squeeze(1)


        # [batch, seq_len, dec_hid_dim] * [batch, dec_hid_dim, 1] -> [batch, seq_len, 1]-> [batch, seq_len]
        attention = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)

        # [batch, seq_len] -> [batch, 1, seq_len]
        attention = F.softmax(attention, dim=1).unsqueeze(1)

        # Compute context vector
        # [batch, 1, seq_len] * [batch, seq_len, dec_hid_dim] -> [batch, 1, dec_hid_dim]
        context = torch.bmm(attention, encoder_outputs)

        return context