import torch
from torch.utils import data
import numpy as np
import nltk

from src.utils import pad_tensor


class yc2Dataset(data.Dataset):
	"""
	Dataset class for loading the feature, target pairs of the YouCook2 dataset together with the corresponding recipe
	type
	"""

	def __init__(self, data, root, vocab, decoder, data_dir, device="cuda", resnet=True):
		self.data = data
		self.root_dir = root
		self.vocab = vocab  
		self.decoder = decoder
		self.resnet = resnet
		self.data_dir = data_dir
		self.device = torch.device(device)

		self.index_pad = vocab.word2idx["<pad>"]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		'''
		Function that returns a triple of frame-wise image features, caption and corresponding recipe type for the
		given video segment
		:param index: index of the video segment
		:return: triple of input features, target and recipe type for the given video segment index
		'''
		vocab = self.vocab

		# load the image features from the numpy file
		img_feature = torch.from_numpy(np.load(self.root_dir + self.data.iloc[index]['video_seg_id'] + '.npy')).float()
		features = img_feature

		# Convert caption (string) to word ids (int) on the basis of the given vocabulary
		caption = self.data.iloc[index]['sentence']
		tokens = nltk.tokenize.word_tokenize(str(caption).lower())
		caption = []
		caption.append(vocab('<start>'))
		caption.extend([vocab(token) for token in tokens])
		caption.append(vocab('<end>'))
		target = torch.Tensor(caption)
		recipe_type = self.data.iloc[index]['recipe_index']

		return features, target, recipe_type


def pad_collate(batch, dim=0, pad_value=0, device=torch.device('cuda')):
	"""
	Function to pad all input sequences to the length of the longest input sequence and the target sequences
	to the length of the longest target sequence. The padded sequences are sorted by the length of the input sequence
	in descending order.

	:param batch: list of triples (feature, target, recipe type)
	:param dim:  dimension to pad
	:param pad_value: value to pad with
	:param device: "cuda" or "cpu"
	:return:
			xs - padded tensor containing the features of all examples in 'batch'
			ys - padded tensor containing the captions of all  examples in 'batch'
			lengths of the input sequences & lengths of the output sequences in 'batch'
			recipe_types -  list of recipe types of all examples in 'batch'
	"""
	max_feature = float(-1)
	max_target = float(-1)
	sequence_lengths = []
	target_lengths = []
	recipe_types = []
	for elem in batch: 
		feature, target, recipe_type = elem
		max_feature = max_feature if max_feature > feature.shape[dim] else feature.shape[dim]
		max_target = max_target if max_target > target.shape[dim] else target.shape[dim]
		sequence_lengths.append(feature.shape[dim])
		target_lengths.append(target.shape[dim])
		recipe_types.append(recipe_type)


	# check for samples that have an input sequence of length zero. Remove those from the batch
	zero_len = [i for i, e in enumerate(sequence_lengths) if e == 0]
	k = 0
	for i in zero_len:
		i = i - k
		k += 1
		del batch[i]

	sequence_lengths = []
	target_lengths = []
	recipe_types = []
	for elem in batch:
		feature, target, recipe_type = elem
		max_feature = max_feature if max_feature > feature.shape[dim] else feature.shape[dim]
		max_target = max_target if max_target > target.shape[dim] else target.shape[dim]
		sequence_lengths.append(feature.shape[dim])
		target_lengths.append(target.shape[dim])
		recipe_types.append(recipe_type)

	sequence_lengths = torch.Tensor(sequence_lengths).to(device)
	target_lengths = torch.Tensor(target_lengths).to(device)
	recipe_types = torch.Tensor(recipe_types).to(device)

	# sort the input sequence lengths
	sequence_lengths, xids = sequence_lengths.sort(descending=True)

	# pad according to the maximum length of the input sequences and maximum length of the target sequences
	batch = [(pad_tensor(x, pad=max_feature, dim=dim, value=0),
			  pad_tensor(y, pad=max_target, dim=dim, value=pad_value))for (x, y, z) in batch]

	# stack all
	xs = torch.stack([x[0] for x in batch], dim=0).to(device,  non_blocking=True)
	ys = torch.stack([x[1] for x in batch]).long().to(device,  non_blocking=True)

	# sort the input and target sequences as well as the corresponding recipe types by the sequence length of the input
	xs = xs[xids].to(device,  non_blocking=True)
	ys = ys[xids].to(device,  non_blocking=True)
	recipe_types = recipe_types[xids]

	return xs, ys, sequence_lengths.int(), target_lengths.int(), recipe_types


def get_loader(root, data, vocab, decoder, batch_size, shuffle, num_workers, data_dir, resnet=False, device="cuda"):
	"""
	Function that build the data loader for the YouCook2 dataset and specified features with a given batch size

	:param root: path to the root directory of the project
	:param data: index lookup dataframe
	:param vocab: vocabulary object with mapping of words to index
	:param decoder: model's decoder
	:param batch_size: batch size for training
	:param shuffle: boolean variable that indicates whether the data should be shuffled during training
	:param num_workers: number of workes used for data loading, Set to 0
	:param data_dir: path to data directory
	:param resnet: 	boolean variable which specifies which vocabulary and embeddings to use. True for using the
					enlarged vocabulary and embeddings of the recipe1M dataset
	:param device:	"cuda" or "cpu"

	:return: data loader with
				image features: a tensor of shape (batch_size, padded length, feature_size(either 512 or 1024).
				captions: a tensor of shape (batch_size, padded_length)
				recipe types: a list of size batch_size
	"""

	# Dataset object for the YouCook2 dataset
	yc2 = yc2Dataset(
		root=root, data=data, vocab=vocab, decoder=decoder, data_dir=data_dir, resnet=resnet, device= device)

	# specify the padding value for the target sequences
	pad_value = yc2.index_pad

	# Build the data loader for a given dataset and batch size
	data_loader = torch.utils.data.DataLoader(
		dataset=yc2, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
		collate_fn=lambda batch: pad_collate(batch, pad_value=pad_value, device=yc2.device))

	return data_loader
