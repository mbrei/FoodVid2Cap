import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.model.attention import Attention
from src.model.mixture_of_experts import MoE

class Decoder(nn.Module):
    '''
    Attention based decoder.
    Decodes encoded visual features of a video with previous word prediction in order to generate captions.
    Architecture is given by:
        ATTENTION - GRU - DROPOUT - MIXTURE-OF-EXPERTS-LAYER (zero-shot approach) - LOG-SOFTMAX
    '''
    def __init__(self, decoder_hid_dem, encoder_hid_dem, vocab, vocab_size, recipe1M_embeddings, embedding_dim,
                 decoder_input_size, linear_input_size, bidirectional_encoder, bidirectional_decoder, dropout,
                 moe=False, num_experts=1, device = torch.device('cuda')):
        '''
        Initialization of the decoder.

        Input:
        :param decoder_hid_dem: integer, hidden dimension of decoder
        :param encoder_hid_dem: integer, hidden dimension of encoder
        :param vocab: used vocabulary of the data
        :param vocab_size: integer, number of words in the vocabulary
        :param recipe1M_embeddings: boolean, True if recipe1M_embedding is considered (for zero-shot approach), else False
        :param embedding_dim: integer, dimension of vocabulary embedding
        :param decoder_input_size: integer, input dimension of decoder
        :param linear_input_size: integer, input dimension of fully connected layer after GRU
        :param bidirectional_encoder: boolean, True if encoder is bidirectional, else False
        :param bidirectional_decoder: boolean, True is decoder is bidirectional, else False
        :param dropout: double, probability of dropout
        :param moe: boolean, True if Mixture of Experts (MoE) layer is considered (for zero-shot approach), else False
        :param num_experts: integer, number of Experts in MoE layer, just for moe=True
        :param device: string, specifies availability of CUDA
        '''
        super(Decoder, self).__init__()
        self.encoder_hid_dem = encoder_hid_dem
        self.decoder_hid_dem = decoder_hid_dem
        self.attention = Attention(encoder_hid_dem, decoder_hid_dem, bidirectional_encoder)
        self.dropout = dropout
        self.output_dim = vocab_size
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.device = device

        # Determine number of directions
        self.decoder_n_direction = 1
        if bidirectional_decoder == True:
            self.decoder_n_direction = 2

        self.bidirectional_encoder = bidirectional_encoder

        self.GRU_layer_out = nn.GRU(decoder_input_size, decoder_hid_dem * 2)
        self.out_layer = nn.Linear(in_features=linear_input_size, out_features=vocab_size)
        self.dropout = nn.Dropout(dropout)

        if recipe1M_embeddings:
            self.embedding_layer, self.embedding_dim = create_recipe1M_embedding_layer(
                torch.Tensor(vocab.weight_matrix), True)
        else:
            self.embedding_layer = nn.Embedding(vocab_size, self.embedding_dim)

        self.moe = moe
        if self.moe:
            self.MoE = MoE(num_experts, decoder_hid_dem, embedding_dim, None, self.decoder_n_direction,
                           device=self.device).to(self.device)

    def forward(self, hidden, actual_word, encoder_outputs, recipe_type):
        '''
        Performs the forward pass of the decoder.

        Inputs:
        :param hidden: torch.tensor, hidden decoder state of previous step
        :param actual_word: torch.tensor, word prediction of previous step
        :param encoder_outputs: torch.tensor, output vector of encoder
        :param recipe_type: torch.tensor, recipe types of samples
        :return: torch.tensor, torch.tensor: softmax probabilities for each word in vocab, hidden decoder state
        '''

        # hidden: [batch_size, dec_hid_dim]
        # actual_word: [batch_size, 1, embedding_dim]
        # encoder_outputs: [batch_size , seq_len, encoder_hid_dim * encoder_n_directional]

        a = self.attention(hidden, encoder_outputs)

        input_char = torch.cat((actual_word, a), 2)

        # [1, batch_size, decoder_input_size]
        input_char = input_char.permute(1, 0, 2)

        # [1 batch_size decoder_hid_dem]
        hidden = hidden.unsqueeze(0)

        # output: [1, batch_size , decoder_n_direction*decoder_hid_dim]
        # hidden: [n_layer*n_direction, batch_size, decoder_hid_dim]
        output, hidden = self.GRU_layer_out(input_char, hidden)

        # Apply Mixture of Experts layer - zero-shot approach
        if self.moe:
            hidden = hidden.to(self.device)
            recipe_type = recipe_type.to(self.device)
            output = self.MoE(hidden.squeeze(dim=0), recipe_type)
            output = torch.cat((output.squeeze(), a.squeeze(1), actual_word.squeeze(1)), dim=1)
            pre_out = self.out_layer(output)
        else:
            output = self.dropout(output)
            output = torch.cat((output.squeeze(0), a.squeeze(1), actual_word.squeeze(1)), dim=1)
            pre_out = self.out_layer(output)

        # prediced_output: [ batch_size , vocab_size ]
        predicted_output = F.log_softmax(pre_out, dim=1)

        return predicted_output, hidden.squeeze(0)


    def add_topic_embedding(self, topic_embedding, non_trainable=False):
        '''
        Add TFIDF-topic embedding

        Input:
        :param topic_embedding: torch.tensor, calculated weights for tfidf topic embedding
        :param non_trainable: boolean, False if embedding will not be optimized during training, else True
        '''
        # Determine dimensions for tfidf topic embedding
        num_embeddings, embedding_dim = topic_embedding.size()

        #Set up embedding with initial tfidf weights
        self.topic_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.topic_embedding.load_state_dict({'weight': topic_embedding})
        if non_trainable:
            self.topic_embedding.weight.requires_grad = False

        # Inform MoE Layer about tfidf embedding
        if self.moe:
            self.MoE = MoE(
                self.num_experts, self.decoder_hid_dem, embedding_dim, self.topic_embedding, self.decoder_n_direction,
                device=self.device).to(self.device)

def create_recipe1M_embedding_layer(weights_matrix, non_trainable=False):
    '''
    Creates Recipe1M vocabulary embedding (for zero-shot approach)

    Input:
    :param weights_matrix: torch.tensor, initial weights for vocabulary embedding
    :param non_trainable: boolean, False if embedding will not be optimized during training, else True

    Output:
    :return: torch.tensor.nn.embedding, integer, emebdding with its dimension
    '''
    # Determine dimensions for vocabulary embedding
    num_embeddings, embedding_dim = weights_matrix.size()

    # Set up emebdding with initial vocabulary weights
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, embedding_dim