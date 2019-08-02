import torch
import torch.nn as nn


class MoE(nn.Module):
    '''
    Applies Mixture of Experts layer. Consisting of primitive experts that are identical in their architecture.
    A gating function learns to manage the experts, such that the experts are enabled to infer captions adjusted to a certain recipe type.
    '''
    def __init__(self, num_experts, input_dim, output_dim, topic_embedding=None, n_direction = 2, device="cuda"):
        '''
        Initialization of the Mixture of Experts Layer.

        Input:
        :param num_experts: integer, number of experts
        :param input_dim: integer, dimension of decoder hidden state
        :param output_dim: integer, tfidf topic embedding dimension
        :param topic_embedding: torch.nn.Embedding, tfidf topic embedding
        :param n_direction: integer, directions of decoder
        :param device: string, specifies availibility of CUDA
        '''
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.experts_list = nn.ModuleList()
        self.topic_embedding = topic_embedding
        self.device = torch.device(device)

        if topic_embedding != None:
            self.multi_perceptron = nn.Linear(self.topic_embedding.weight.size()[1], 1)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)

        # Create a list of identical experts
        for i in range(num_experts):
            self.experts_list.append(Expert(input_dim * n_direction, output_dim * n_direction))


    def forward(self, hidden_decoder_state, recipe_types):
        '''
        Performs forward pass of the MoE layer

        Input:
        :param hidden_decoder_state: torch.tensor, hidden state of the decoder
        :param recipe_types: torch.tensor, recipe types of batch samples

        Output:
        :return: torch.tensor, output after applying MoE layer
        '''
        batch_size = hidden_decoder_state.size()[0]
        exp_output_list = []

        # Apply each expert on the hidden decoder state
        for i, expert in enumerate(self.experts_list):
            exp_output_list.append(expert(hidden_decoder_state))

        # [num_experts, batch_size, hidden_decoder_dim]
        experts_output = torch.stack(exp_output_list)
        experts_output = experts_output.permute(1, 2, 0).to(self.device)

        # Gating function learns from the tfidf-topic embedding
        if self.topic_embedding == None:
            weights = torch.randn(batch_size, self.num_experts, 1)
        else:
            weights = torch.zeros(batch_size, self.topic_embedding.weight.size()[1])
            for i, rt in enumerate(recipe_types):
                weights[i, :] = self.topic_embedding(rt.long())

            weights = weights.to(self.device)
            weights = self.multi_perceptron(weights)
            weights = self.relu(weights)
            weights = self.softmax(weights)
            weights = weights.repeat(self.num_experts, 1, 1)
            weights = weights.permute(1, 0, 2)

        weights = weights.to(self.device)
        output = torch.bmm(experts_output, weights)
        output = output.squeeze(dim=2)
        return output


class Expert(nn.Module):
    '''
    Creates a single expert for the MoE layer.
    '''
    def __init__(self, input_dim, output_dim):
        '''
        Initalize primitive expert.

        :param input_dim: integer, input dimension
        :param output_dim: integer, output dimension
        '''
        super(Expert, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, hidden_decoder_state):
        '''
        Forward pass of expert.

        :param hidden_decoder_state: torch.tensor, hidden state of the decoder
        :return: torch.tensor, output of applied expert
        '''
        output = self.linear(hidden_decoder_state)
        output = self.relu(output)

        return output

