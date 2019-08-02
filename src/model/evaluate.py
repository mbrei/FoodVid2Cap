import torch
from nltk.translate.meteor_score import single_meteor_score



class Evaluate(object):

    def __init__(self):
        """
        Construct a new Evaluate instance.
        Initialize the list of candidate and reference sentences and the initial meteor score.
        """
        self.candidates = []
        self.references = []
        self.meteor_score = 0


    def evaluate_iteration(self, prediction_probs, captions, vocab):
        '''
        Calculates the METEOR scores for the predicted feature vectors.

        Input:
        :param prediction_probs: torch.tensor filled with predictions of the seq-to-seq model
        :param captions: torch.tensor filled with the ground truth captions
        :param vocab: used vocabulary of model
        '''
        values, indices = prediction_probs.max(2)
        words = []
        # Find corresponding word in the vocabulary
        for y in range(indices.size()[1]):
            words.append([vocab.idx2word[int(word_idx)] for word_idx in indices[:, y]])

        k = prediction_probs.size()[1]

        # Calculate meteor score for every sample in batch
        meteor = 0
        for i in range(k):

            words_true = [vocab.idx2word[int(word_idx)] for word_idx in captions[i]]
            words_pred = words[i]

            try:
                end_index_pred = words_pred.index("<end>")
            except ValueError:
                end_index_pred = len(words_pred)

            end_index_true = words_true.index("<end>")

            self.references.append([words_true[1:end_index_true]])
            self.candidates.append(words_pred[1:end_index_pred])

            sen_true = ' '.join(words_true[1:end_index_true])
            sen_pred = ' '.join(words_pred[1:end_index_pred])

            meteor_score_single = single_meteor_score(sen_true, sen_pred)
            meteor += meteor_score_single

        self.meteor_score += meteor/k









