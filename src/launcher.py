"""
Script for training the Zero-Shot model which is based on a sequence-to-sequence model with attention mechanism.
The following arguments can be set for the training process:

--recipe1M                  boolean variable which specifies which vocabulary and embeddings to use.
                            True for using the enlarged vocabulary and embeddings of the recipe1M dataset
--resNet                    boolean varibale which specifies which features to use. *True* for using the ResNet features
--moe                       boolean varibale which specifies whether the model should be trained with
                            the Mixture of Experts (MoE) layer
--num_experts               specifies the number of experts for the MoE layer
--num_epochs                specifies the number of epochs for training the model
--print_every               specifies the frequency of printing the training and validaiton results and saving the model
--learning_rate             specifies the learning rate for the Adam optimizer
--batch_size                specifies the batch size for each iteration
--teaching_forcing_ratio    During training, the teacher forcing ratio can be between 0 and 1. At inference time it has
                            to be set to 0
--limit                     specifies the length of the generated output sequence. If set to zero the length equals to
                            the maximum length of the annotations in the training dataset
--num_train                 specifies the number of samples in training
--num_val                   specifies the number of samples in validation
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import sys
import random
import pickle
import os
from datetime import datetime
import json
from scipy.sparse import load_npz

# specify the root path of the project for importing own modules
sys.path.append("C:/Users/User/foodcap")

# misc
import src.config as config
from src.data.data_loading import get_loader
from src.data.load_build_vocab import Vocabulary
from src.model.encoder import EncoderRNN
from src.model.decoder import Decoder
from src.model.Seq2Seq import Seq2Seq
from src.model.trainer import train
from src.model.topic_embedding import TFIDF
from src.utils import save_list_to_file, save_model

# parse arguments
parser = ArgumentParser()

parser.add_argument('-e', '--num_epochs', default=20,
                    dest="num_epochs", type=int)
parser.add_argument('-p', '--print_every', default=5,
                    dest="print_every", type=int)
parser.add_argument('-nT', "--num_train", default=1000,
                    dest="num_train", type=int)
parser.add_argument('-nV', "--num_val", default=20,
                    dest="num_val", type=int)
parser.add_argument('-lr', "--learning_rate", default=0.001,
                    dest="learning_rate", type=float)
parser.add_argument('-bs', "--batch_size", default=64,
                    dest="batch_size", type=int)
parser.add_argument('-worker', "--num_workers", default=0,
                    dest="num_workers", type=float)
parser.add_argument('-ex', "--num_experts", default=8,
                    dest="num_experts", type=int)
parser.add_argument('-tfr', "--teaching_forcing_ratio", default=0.8,
                    dest="teaching_forcing_ratio", type=float)
parser.add_argument('-limit', "--limit", default=0,
                    dest="limit", type=int)
parser.add_argument('--recipe1M', dest='recipe1M', action='store_true')
parser.add_argument('--moe', dest='moe', action='store_true')
parser.add_argument('--resNet', dest='resNet', action='store_true')

parser.set_defaults(recipe1M=False)
parser.set_defaults(moe=False)
parser.set_defaults(resNet=True)

args = parser.parse_args()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

# paths
data_dir = config.DATA["data_dir"]

if args.resNet:
    feature_dir = "../data/resnet_features/"
else:
    feature_dir = '../data/bn_features/'              # directory for image features

tfidf_weight_matrix_path_original = data_dir + 'tfidf_weight_matrix.npz'
tfidf_weight_matrix_path_processed = data_dir + 'tfidf_weight_matrix_correct.npy'
recipe_label_index_path = data_dir + 'label_index_foodtype.csv'

save_model_path = os.path.join(
    "../models/",
    datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(save_model_path)

recipe1M_embeddings = args.recipe1M

if recipe1M_embeddings:
    vocab_path = '../data/vocab_recipe1M.pkl'
else:
    vocab_path = "../data/vocab.pkl"

# model parameters and specifications
if args.resNet:
    in_features = 512
else:
    in_features = 1024

hidden_size_encoder = 300
num_layers_encoder = 1
bidirectional_encoder = True
num_dnn_layers_encoder = 1
dropout_encoder = 0.1
encoder_output_size = 300

num_experts = args.num_experts
embedding_size = 300
hidden_size_decoder = 300
num_layers_decoder = 1
bidirectional_decoder = True
dropout_decoder = 0.1

num_dir_enc = 2 if bidirectional_encoder else 1
num_dir_dec = 2 if bidirectional_decoder else 1

decoder_input_size = hidden_size_encoder *num_dir_enc+ embedding_size
linear_input_size = decoder_input_size + hidden_size_decoder*2

# training parameters
num_epochs = args.num_epochs
print_every = args.print_every
learning_rate = args.learning_rate
batch_size = args.batch_size
num_workers = args.num_workers

teaching_forcing_ratio = args.teaching_forcing_ratio
limit = args.limit
limit_validation = 25

num_train = args.num_train
num_val = args.num_val

# data loading - load vocabulary and index lookup data file
print("Some data loading...")

print("Load vocabulary")
# Load vocabulary wrapper
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
print("Vocab size", vocab_size)

print("Load Data")
yc2_all = pd.read_csv("../data/data_all.csv")

train_idx = yc2_all[yc2_all.subset_new == "training"].index
val_idx = yc2_all[yc2_all.subset_new == "validation"].index
test_idx = yc2_all[yc2_all.subset_new == "test"].index

train_idx = random.sample(list(train_idx), num_train)
val_idx = random.sample(list(val_idx), num_val)

yc2_train = yc2_all.iloc[train_idx]
yc2_val = yc2_all.iloc[val_idx]

yc2_train = yc2_train.reset_index().drop('level_0', axis=1)
yc2_val = yc2_val.reset_index().drop('level_0', axis=1)

yc2_train_small = yc2_train[['video_seg_id', 'recipe_label', 'sentence', 'recipe_index']]
yc2_val_small = yc2_val[['video_seg_id', 'recipe_label', 'sentence', 'recipe_index']]

del yc2_all
del yc2_train
del yc2_val

print("Done!")


# Build Seq2Seq Model
encoder = EncoderRNN(num_features=in_features, hidden_size=hidden_size_encoder, num_layers=num_layers_encoder,
                     num_dnn_layers=num_dnn_layers_encoder, dropout=dropout_encoder,
                     bidirectional=bidirectional_encoder, device=device).to(device)

decoder = Decoder(decoder_hid_dem=hidden_size_decoder, encoder_hid_dem=hidden_size_encoder, vocab=vocab,
                  vocab_size=vocab_size, recipe1M_embeddings=recipe1M_embeddings, embedding_dim=embedding_size,
                  decoder_input_size=decoder_input_size, linear_input_size=linear_input_size,
                  bidirectional_encoder=bidirectional_encoder, bidirectional_decoder=bidirectional_decoder,
                  dropout=dropout_decoder, moe=args.moe, num_experts=num_experts, device=device).to(device)

model = Seq2Seq(encoder, decoder, device).to(device)

# if the model should be trained with the MoE layer, then the TFIDF based topic embeddings
# for the different recipe types are added to the decoder
if args.moe:

    with open(data_dir + 'R1M_vocab.json', 'rb') as f:
        R1M_vocab_names = json.load(f)

    tfidf_weight_matrix = load_npz(tfidf_weight_matrix_path_original)
    recipe_labels = pd.read_csv(recipe_label_index_path)

    tfidf = TFIDF(decoder, vocab, recipe_labels, R1M_vocab_names, device)
    topic_embedding = tfidf.get_w_tfidf_via_names(tfidf_weight_matrix)
    decoder.add_topic_embedding(topic_embedding, non_trainable=True)

    if use_cuda: 
        np.save(tfidf_weight_matrix_path_processed,topic_embedding.to('cpu').data.numpy())
    else: 
        np.save(tfidf_weight_matrix_path_processed,topic_embedding.data.numpy())

    # once the tfidf_weight_matrix_correct is created and saved to the data directory, the above code does not have to
    # be executed again. It is then sufficient to execute the following line of code
    # decoder.add_topic_embedding(torch.tensor(np.load(tfidf_weight_matrix_path_processed),).float(), non_trainable=True)

# Loss and optimizer
params = list(decoder.parameters()) + list(encoder.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

PAD_IDX = vocab.word2idx['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# Build  data loader for the training and validation dataset
data_loader_train = get_loader(feature_dir, yc2_train_small, vocab, decoder, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, data_dir=data_dir, resnet=args.resNet, device=device)
data_loader_val = get_loader(feature_dir, yc2_val_small, vocab, decoder, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, data_dir=data_dir, resnet=args.resNet, device=device)

if args.moe:
    decoder.add_topic_embedding(torch.tensor(np.load(tfidf_weight_matrix_path),).float(), non_trainable=True)

'''
TRAINING

During training the training loss is as well as the evalution scores for training and evaluation are saved.
Furthermore, if the model achieves a better value for one of the evaluation scores on the validations set, the model is 
saved. Also, after a defined number of epochs (set with print_every) the model is saved and the values for the training 
loss and METEOR scores on the training and vaidation set are reported
'''
losses_train = []
bleus_1_train = []
bleus_2_train = []
bleus_3_train = []
bleus_4_train = []
meteors_train = []
cider_train = []

bleus_1_val = []
bleus_2_val = []
bleus_3_val = []
bleus_4_val = []
meteors_val = []

meteor_best = 0
meteor_train_best = 0
b1_best = 0
b2_best = 0
b3_best = 0
b4_best = 0

print("Start training")
for epoch in range(num_epochs):
    e_loss_train, bleu_1_train, bleu_2_train, bleu_3_train, bleu_4_train, meteor_score_train, seq_probs_train,\
    captions_train = train(model=model, data_loader=data_loader_train, vocab=vocab, optimizer=optimizer,
                           criterion=criterion, teaching_forcing_ratio=teaching_forcing_ratio, limit=limit,
                           train=True)

    losses_train.append(e_loss_train)
    bleus_1_train.append(bleu_1_train)
    bleus_2_train.append(bleu_2_train)
    bleus_3_train.append(bleu_3_train)
    bleus_4_train.append(bleu_4_train)
    meteors_train.append(meteor_score_train)

    # save training loss and evaluation scores of training
    save_list_to_file(losses_train, save_model_path, 'loss_train_list.json')
    save_list_to_file(meteors_train, save_model_path, 'meteor_train_list.json')
    save_list_to_file(bleus_1_train, save_model_path, 'bleus1_train_list.json')
    save_list_to_file(bleus_2_train, save_model_path, 'bleus2_train_list.json')
    save_list_to_file(bleus_3_train, save_model_path, 'bleus3_train_list.json')
    save_list_to_file(bleus_4_train, save_model_path, 'bleus4_train_list.json')

    if device == torch.device("cuda:0"):
        torch.cuda.empty_cache()

    e_loss_val, bleu_1_val, bleu_2_val, bleu_3_val, bleu_4_val, meteor_score_val, seq_probs_val, captions_val = \
        train(model=model, data_loader=data_loader_val, vocab=vocab, optimizer=optimizer, criterion=criterion,
              teaching_forcing_ratio=0, limit=limit_validation, train=False)

    bleus_1_val.append(bleu_1_val)
    bleus_2_val.append(bleu_2_val)
    bleus_3_val.append(bleu_3_val)
    bleus_4_val.append(bleu_4_val)
    meteors_val.append(meteor_score_val)

    # save evaluation scores of the validation
    save_list_to_file(meteors_val, save_model_path, 'meteor_val_list.json')
    save_list_to_file(bleus_1_val, save_model_path, 'bleus1_val_list.json')
    save_list_to_file(bleus_2_val, save_model_path, 'bleus2_val_list.json')
    save_list_to_file(bleus_3_val, save_model_path, 'bleus3_val_list.json')
    save_list_to_file(bleus_4_val, save_model_path, 'bleus4_val_list.json')

    # save model if model achieves better evaluation scores on the validation set
    meteor_best = save_model(reference_value=meteor_best, candidate_value=meteor_score_val, model=model,
                             path=save_model_path, model_name='model_val_meteor')
    meteor_train_best = save_model(reference_value=meteor_train_best, candidate_value=meteor_score_train, model=model,
                                   path=save_model_path, model_name='model_train_meteor')
    save_model(reference_value=b1_best, candidate_value=bleu_1_val, model=model, path=save_model_path,
               model_name='model_bleu_1')
    save_model(reference_value=b2_best, candidate_value=bleu_2_val, model=model, path=save_model_path,
               model_name='model_bleu_2')
    save_model(reference_value=b3_best, candidate_value=bleu_3_val, model=model, path=save_model_path,
               model_name='model_bleu_3')
    save_model(reference_value=b4_best, candidate_value=bleu_4_val, model=model, path=save_model_path,
               model_name='model_bleu_4')

    if epoch % print_every == 0:
        torch.save(model, save_model_path + '/model_' + str(epoch))
        print("Epoch:", epoch)

        print("Train")
        print("Loss:", e_loss_train)
        print("METEOR score: %f" % meteor_score_train)

        print("Validation")
        print("METEOR score: %f" % meteor_score_val)

        print("_______________________________")

    if use_cuda:
        torch.cuda.empty_cache()

# save last model after finishing training process
torch.save(model, save_model_path+"/last_model")
