import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    '''
    Applies Seq-to-seq model in video captioing tasks with attention mechanism.
    '''
    def __init__(self, encoder, decoder, device):
        '''
        Initialization of the Seq-to-Seq model.

        Input:
        :param encoder: torch.nn.Module, encoder
        :param decoder: torch.nn.Module, decoder
        :param device: string, specifies availibility of CUDA
        '''

        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, vocab, vid_feats, video_lengths, captions, recipe_type, teaching_forcing_ratio, train = True, limit=0):
        '''
        Performs forward pass of Seq-to-Seq model. Distinguishes between training and inference time.

        Input:
        :param vocab: used vocabulary of the data
        :param vid_feats: torch.tensor, visual features of the batch samples
        :param video_lengths: torch.tensor, list of video lengths of the batch samples
        :param captions: torch.tensor, captions of the batch samples
        :param recipe_type: torch.tensor, list of recipe types of batch samples
        :param teaching_forcing_ratio: double, if non-zero probability of using teacher forcing at training time, else no teacher forcing is used
        :param train: boolean, True if model is in training, False if model is in inference
        :param limit: integer, max length of caption to be considered

        Output:
        :return: torch.tensor, sequence of predicted words
        '''

        batch_size = vid_feats.size()[0]
        if (limit == 0):
            max_len = captions.size()[1]
        else:
            max_len = limit

        vocabsize = self.decoder.output_dim

        # First word is always SOS
        SOS_index = vocab.word2idx["<start>"]
        actual_word = self.decoder.embedding_layer(torch.tensor(SOS_index).view(1, -1)
                                                   .to(self.device)).repeat(batch_size, 1, 1)

        # Encode visual features
        encoder_outputs, hidden = self.encoder(vid_feats, video_lengths)
        encoder_outputs, hidden = encoder_outputs.to(self.device), hidden.to(self.device)

        # Initialize tensor for predictions
        predicted_word = torch.zeros(max_len, batch_size, vocabsize).to(self.device)
        predicted_word[0, :, SOS_index] = 1

        # prediction when model mode is 'training'
        if train:
            for t in range(1, max_len):
                # Decode visual features with
                output, hidden = self.decoder(hidden, actual_word, encoder_outputs, recipe_type)
                output, hidden = output.to(self.device), hidden.to(self.device)
                predicted_word[t] = output
                # Greedy search for best predicted value with corresponding index
                topv, topi = output.topk(1)
                bs = topi.size()[0]     #batch_size
                temp2 = torch.zeros(0, 1, self.decoder.embedding_dim).to(self.device) #empty tensor for predictions at step t
                for row in range(bs):
                    index = topi[row][0].item()
                    temp = self.decoder.embedding_layer(torch.tensor(index).view(1, -1).to(self.device))
                    temp2 = torch.cat((temp2, temp))

                # Use teacher forcing, determine randomly if ground truth or prediction is used in next prediction step
                teacher_force = random.random() < teaching_forcing_ratio
                if teacher_force == True:
                    actual_word = self.decoder.embedding_layer(captions[:, t]).unsqueeze(1)
                else:
                    actual_word = temp2

        # prediction when model mode is 'inference'
        else:
            for t in range(1, max_len):
                output, hidden = self.decoder(hidden, actual_word, encoder_outputs, recipe_type)
                output, hidden = output.to(self.device), hidden.to(self.device)
                predicted_word[t] = output
                topv, topi = output.topk(1)
                bs = topi.size()[0]     #batch_size
                temp2 = torch.zeros(0, 1, self.decoder.embedding_dim).to(self.device)
                for row in range(bs):
                    index = topi[row][0].item()
                    temp = self.decoder.embedding_layer(torch.tensor(index).view(1, -1).to(self.device))
                    temp2 = torch.cat((temp2, temp))
                # Just actual prediction of model is allowed for using in next prediction step
                actual_word = temp2

        return predicted_word