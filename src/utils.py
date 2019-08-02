"""
Script that contains some util functions that are used across the project
"""

import torch
import pandas as pd
import json
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge


def pad_tensor(vec, pad, value=0, dim=0):
    """
    Function that pads a tensor with a given value to a given size
    :param vec: tensor to pad
    :param pad: the size to pad to
    :param value: value to pad with
    :param dim: dimension to pad
    :return: a new tensor padded with the 'value' to size 'pad' in dimension 'dim'
    """
    pad_size = pad - vec.shape[0]

    if len(vec.shape) == 2:
        zeros = torch.ones((pad_size, vec.shape[-1])) * value
    elif len(vec.shape) == 1:
        zeros = torch.ones((pad_size,)) * value
    else:
        raise NotImplementedError
    return torch.cat([torch.Tensor(vec), zeros], dim=dim)


def load_json_data(path):
    """
    Function that loads the annotations data from the YouCook2 dataset from a json file and
    returns it as pandas dataframe
    :param path: path to the json file
    :return: pandas datframe that contains the loaded data from the json file
    """
    df = pd.read_json(path)
    df = df.reset_index()
    database = pd.DataFrame(df.database.values.tolist())
    df = df.drop(['database'], axis=1)
    df = df.join(database)

    return df


def save_model(reference_value, candidate_value, model, path, model_name):
    """
    Function to save a torch model when the current evaluation score is higher then the reference value.
    If so the model is saved to a given path with a specific name and the reference value is updated and returned.
    If not, the old reference value is returned.

    :param reference_value:
    :param candidate_value:
    :param model: Model to save
    :param path: Path where the model should be saved
    :param model_name: name for the model to be saved with
    :return: reference value; equals candidate value, if model achieves a higher evaluation score, old reference value
             otherwise
    """
    if reference_value < candidate_value:
        reference_value = candidate_value
        torch.save(model, path+"/"+model_name)

    return reference_value


def save_list_to_file(list_, path, file):
    """
    Function to save the values from a list to a file in a specific path
    :param list_: list to save
    :param path: path where the file should be savec
    :param file: file name
    :return:
    """
    with open(path+"/"+file, 'w') as f:
        f.write(json.dumps(list_))


def create_df(k, vocab, seq_probs, captions):
    """
    Function to create a dictionary that contains the predicted sentences that arise from the predicted probabilities.
    Furthermore, the ground truth sentences are stored.

    :param k: number of samples
    :param vocab: Vocabulary to lookup the words that correspond to the predicted indices
    :param seq_probs: Predicted probabilities from the captioning model
    :param captions: ground truth in index form
    :return: dictonary with predicted and ground truth sentences
    """
    dic_result = []
    values, indices = seq_probs.max(2)
    words = []
    for y in range(indices.size()[1]):
        words.append([vocab.idx2word[int(word_idx)] for word_idx in indices[:, y]])

    for i in range(k - 1):
        result_segment = {'Ground_truth': '', 'Predicted': ''}
        words_true = [vocab.idx2word[int(word_idx)] for word_idx in captions[i]]
        words_pred = words[i]

        result_segment["Predicted"] = ' '.join(words_pred)
        result_segment["Ground_truth"] = ' '.join(words_true)
        dic_result.append(result_segment)

    return dic_result


def calculate_scores(df):
    """
    Function to perform a row-wise calculation of the evaluation scores.
    The calculated scores are the METEOR score and ROUGE-L score.

    :param df: row of a dataframe with columns 'prediction' and 'ground_truth'
    :return: row of dataframe with calculated scores
    """
    rouge = Rouge()

    pred = df.prediction
    sen = df.ground_truth

    meteor = single_meteor_score(pred, sen)
    scores_rouge = rouge.get_scores(pred, sen)

    df["meteor"] = meteor
    df["rouge-l"] = scores_rouge[0]["rouge-1"]["f"]

    return df


def create_prediction_df(json_pred_path, data, type):
    """
    Function create a dataframe that maps the predicted sentences to their ground_truth and index of the video segment.

    :param json_pred_path:  Path to the JSON file of the dictionary that contains the predicted sentences and
                            the corresponding ground truth
    :param data:            index lookup file for all video segments
    :param type:            specifiy whether the predictions are from the test or validation set
    :return:                panadas dataframe that maps the predicted caption together with video segment index,
                            the recipe type and its groud truth sentence

    """

    if type == "test":
        data_val = data[data.recipe_type.isin([201, 122, 313, 101, 229, 202, 412])]

    if type == "val":
        data_val = data[data.recipe_type.isin([215, 107, 216, 305, 403, 126, 211])]

    with open(json_pred_path, 'r') as f:
        preds_val = json.load(f)

    preds = []
    start = '<start>'
    end = '<end>'

    videos = data_val["index"].unique()

    for idx in videos:

        data_video = data[data["index"] == idx]

        for i, d in data_video.iterrows():

            video_seg_id = d.video_seg_id
            sen = d.sentence

            # look for the corresponding ground truth sentence
            for pred in preds_val:

                sen_g = pred["Ground_truth"]
                gt = sen_g[sen_g.find(start) + len(start) + 1:sen_g.find(end) - 1]

                if gt == sen:
                    sen_p = pred["Predicted"]
                    pred = sen_p[sen_p.find(start) + len(start) + 1:sen_p.find(end)]

                    dm = {"video_index": idx, "video_seg_id": video_seg_id, "ground_truth": sen, "prediction": pred,
                          "recipe": data_video["recipe_label"].iloc[0]}
                    preds.append(dm)

    df = pd.DataFrame(preds)

    return df


class WordCounter:
    """
    Class for counting the words from the predicted sentences and ground truth captions for each recipe type
    in the test set separately
    """

    def __init__(self):
        self.recipe_types = []

    def count_words(self, df, column):
        """
        Function to count the words of a dataframes column. The column must contain a string variable

        :param df: pandas dataframe for which the words should be counted
        :param column:  name of the column that contains a string variable (prediction or ground_truth sentences)
                        for which the words should be calculated
        :return:
        """
        counts= (df[column].str.split(expand=True)
               .stack()
               .value_counts()
               .rename_axis('vals')
               .reset_index(name='count'))
        counts["recipe"] = df.recipe.iloc[0]
        counts["type"] = column

        # The counts are appended to a global list that stores all counts for all of the recipe types
        self.recipe_types.append(counts)

    def count_words_per_recipe(self, df):
        """
        Function that counts the words of the predicted and ground truth sentences for each recipe type separately.
        Furthermore, the ratio of occurence for each word in the predicted and ground truth sentences is calculated.

        :param df:  pandas dataframe with prediction and ground truth sentences
        :return:    new pandas dataframe that holds the total count and ratio for each word in the predicted
                    and ground truth sentences for each recipe type separately
        """

        # count the words in the predicted and ground truth sentecnes
        df.groupby("recipe").apply(lambda x: self.count_words(x, "prediction"))
        df.groupby("recipe").apply(lambda x: self.count_words(x, "ground_truth"))

        words = pd.concat(self.recipe_types)

        # exclude words that appear frequently across all recipe types from the analysis
        words_red = words[~words.vals.isin(["the", "and", "on", "a", "to", "in", "of", "it", "add", "with", "into",
                                            "some", "place", "pan", "for"])]
        words_red = words_red.drop_duplicates()

        # sum up the word counts for each recipe type separately
        agg = words_red.groupby(["recipe", "type"])["count"].agg("sum")
        words_red = words_red.merge(agg, left_on=["recipe", "type"], right_index=True)

        # calculate the ratio of occurence for a word in a recipe type
        words_red["ratio"] = words_red["count_x"] / words_red["count_y"]
        words_red.columns = ["word", "count", "recipe", "type", "sum_count", "ratio"]

        return words_red

