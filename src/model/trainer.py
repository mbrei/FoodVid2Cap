import torch
import torch.utils.data
from nltk.translate.bleu_score import corpus_bleu

from src.model.evaluate import Evaluate

def train(model, data_loader, vocab, optimizer, criterion, teaching_forcing_ratio, limit, train =True, device='cuda'):
    '''
    Executes the training of a seq-to-seq model.

    Input:
    :param model: torch.nn.Module, seq-to-seq model
    :param data_loader: torch.utils.data, data_loader
    :param vocab: used vocabulary of the model
    :param optimizer: torch.optim, desired optimizer
    :param criterion: torch.nn, loss function
    :param teaching_forcing_ratio: double, probability of using teacher forcing
    :param limit: integer, maximal length of captions to be considered
    :param train: TRUE if model is in training, else FALSE
    :param device: string, specifies availability of CUDA

    Output:
    :return: integer(6), torch.tensor(2), epoch loss, scores (BLEUI 1-4, METEOR), predictions of model, captions
    '''
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0
    Evaluater = Evaluate()

    # Loop through dataset, executed on single batch
    for i, (x_train, captions, x_lengths_train, y_lengths_train, x_recipe_type) in enumerate(data_loader):
        
        optimizer.zero_grad()

        # Get predictions
        seq_probs= model(vocab=vocab, vid_feats=x_train, video_lengths=x_lengths_train, captions=captions,
                         recipe_type=x_recipe_type, teaching_forcing_ratio=teaching_forcing_ratio, train=True,
                         limit=limit)

        # Evaluate
        Evaluater.evaluate_iteration(seq_probs, captions, vocab)

        # Just for training: Loss calculation + optimization of model
        if train:
            target = captions.permute(1, 0)
            output = seq_probs[1:].view(-1, seq_probs.shape[-1])
            trg = target[1:].contiguous().view(-1)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()

        if train:
         epoch_loss += loss.item()

    # Calculate epoch loss
    num_iter = len(data_loader)
    epoch_loss = epoch_loss/num_iter

    # Calculate scores: METEOR, BLEU 1-4
    meteor_score = Evaluater.meteor_score/num_iter
    bleu_1 = corpus_bleu(Evaluater.references, Evaluater.candidates, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(Evaluater.references, Evaluater.candidates, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(Evaluater.references, Evaluater.candidates, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(Evaluater.references, Evaluater.candidates, weights=(0.25, 0.25, 0.25, 0.25))

    return epoch_loss, bleu_1, bleu_2, bleu_3, bleu_4, meteor_score, seq_probs, captions
