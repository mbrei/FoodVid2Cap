import torch

class TFIDF:
	'''
		Calculates the tfidf weights for each recipe type and each word of the vocabulary
	'''
	def __init__(self, decoder, vocab, recipe_types, vocab_names, device):
		'''
		Initialize the tfidf weight calculator

		Input:
		:param decoder: torch.nn.Module, decoder of seq-to-seq model
		:param vocab: used vocabulary of the data
		:param recipe_types: pandas dataframe, recipe labels with index
		:param vocab_names: list of all words that have been appeared in recipe1M recipes
		:param device: string, specifies availability of CUDA
		'''

		self.decoder = decoder
		self.recipe_types = recipe_types
		self.device = device
		self.vocab = vocab
		self.vocab_names = vocab_names
		self.embedding_size = decoder.embedding_layer.embedding_dim

	def get_w_tfidf_via_names(self, tfidf_weights_R1M):
		'''
		Calculates tfidf weights for each topic.

		Input:
		:param tfidf_weights_R1M: tfidf weights calculated on the entire recipe instructions of Recipe1Million

		Output:
		:return: torch.tensor, tfidf weight matrix adjusted to vocab
		'''
		self.w_tfidf = torch.zeros(len(self.recipe_types), self.embedding_size).to(self.device)

		# Loop through all possible recipe_types (topics)
		for i in range(len(self.recipe_types)):
			recipe_index = self.recipe_types.loc[i]['recipe_index']
			words_embed = torch.zeros(len(self.vocab_names), self.embedding_size).to(self.device)

			# Loop through all possible new words to determine product of tfidf-topic-weight and word-embedding-vector
			for j, name in enumerate(self.vocab_names):
				if name in self.vocab.word2idx.keys():
					word_index = self.vocab.word2idx[name]
					word_emb = self.decoder.embedding_layer(torch.tensor(word_index).view(1, -1).to(self.device)).squeeze(
						dim=0).to(self.device)

					words_embed[j, :] = word_emb * torch.tensor(tfidf_weights_R1M[i, j].item()).float().to(self.device)
			sum_ = torch.sum(words_embed, dim=0).to(self.device)
			self.w_tfidf[recipe_index, :] = sum_
		return self.w_tfidf


