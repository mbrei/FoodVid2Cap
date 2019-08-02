"""
Script for building the video features for the relevant segments of the video sequences and
storing them as numpy files.
Additionally, an index lookup dataframe is built that holds the references to the feature files together with the
corresponding recipe types. Furthermore, this dataframe stores the belonging of each video sequence to
the train, validation or test dataset.
The video sequences are split such that the 89 recipes are divided to train, validation and test set at
a ratio of 75, 7 and 7. This specific data split ensures the Zero-Shot setting.

Arguments:
--resNet:   build the video features on the basis of the provided ResNet features from
            the YouCook2 dataset(http://youcook2.eecs.umich.edu/)
--bn:      build the video features on the basis of the provided motion features(with suffix _bn.npy)
            from DenseCap (https://github.com/LuoweiZhou/densecap)
"""

import numpy as np
import math
import pandas as pd
import os
from tqdm import tqdm
import sys
import argparse

# specify the root path of the project for importing own modules
sys.path.append("C:/Users/User/foodcap")

import src.config as config
from src.utils import load_json_data


def get_resnet_feature(data, index, feature_dir, save_dir):
    """
    Function for retrieving the provided ResNet features from the YouCook2 dataset for a specific video segment
    and storing them as numpy files.
    Independent of the duration, each video is sampled with 500 frames. To obtain the feature vector for a specific
    segment of a video, this function extracts the frame-wise features based on the segment start and end.
    :param data: lookup dataframe with video segment index as index
    :param index: index of the video segment
    :param feature_dir: path to the ResNet features
    :param save_dir: path for saving the feature files as numpy files
    :return:
    """

    if data.iloc[index]['subset'] == 'training':
        path = 'train_frame_feat_csv/'
    elif data.iloc[index]['subset'] == 'validation':
        path = 'val_frame_feat_csv/'
    path += str(data.iloc[index]['recipe_type'])+'/'+ str(data.iloc[index]['index'])+'/0001/resnet_34_feat_mscoco.csv'
    resnet_feature = np.genfromtxt(feature_dir +path, delimiter=',')
    ''' for the given segment extract the relevant frames of the video based on the start and end second of the segment 
     by recalculating the segment end and start to the number of 500 frames '''
    total_duration = data.iloc[index]['duration']
    interval_duration = float(float(total_duration) / 500)
    seg_start = math.ceil(int(data.iloc[index]['seg_start'])/interval_duration)
    seg_end = math.ceil(int(data.iloc[index]['seg_end'])/interval_duration)
    img_feature = resnet_feature[seg_start:seg_end+1]
    # save the frame-wise features as numpy file
    np.save(save_dir+str(data.iloc[index]['video_seg_id']+'.npy'), img_feature)


def get_bn_feature(data, index, feature_dir, save_dir):
    """
     Function for retrieving the provided motion features from DenseCap for a specific video segment
     and storing them as numpy files.
     Each video is sampled every 0.5 second. To obtain the feature vector for a specific
     segment of a video, this function extracts the frame-wise features based on the segment start and end.
     :param data: lookup dataframe with video segment index as index
     :param index: index of the video segment
     :param feature_dir: path to the features
     :param save_dir: path for saving the feature files as numpy files
     :return:
     """
    # get start and end second of the given video segment and recalculate them to the sampling rate of 0.5 seconds
    seg_start = 2 * int(data.iloc[index]['seg_start'])
    seg_end = 2 * (int(data.iloc[index]['seg_end']) + 1)

    # load the frame-wise motion features from DenseCap
    # The relevant frames are indicated by the segment end and start
    bn_feature = np.load(os.path.join(
        feature_dir, data.iloc[index]['index'] + '_bn.npy'))[seg_start:seg_end]

    # save the frame-wise features as numpy file
    np.save(save_dir + str(data.iloc[index]['video_seg_id'] + '.npy'), bn_feature)


def create_index_data_file(data_dir):
    """

    :param data_dir: path to the data directory
    :return: index lookup dataframe that holds the references to the feature files together with the
    corresponding recipe types. Furthermore, this dataframe stores the belonging of each video sequence to
    the train, validation or test dataset.
    """

    # read in the recipe labels of the YouCook2 dataset
    recipe_labels = pd.read_csv(data_dir + config.DATA["recipe_label_path"])
    recipe_labels = recipe_labels.T.reset_index().T
    recipe_labels = recipe_labels.to_dict(orient='records')
    recipe_labels_dic = {}
    for i in recipe_labels:
        recipe_labels_dic[int(i[0])] = i[1]
    recipe_types = list(recipe_labels_dic.values())

    # load the annotations of the YouCook2 dataset
    annotations_path = data_dir + config.DATA["annotations_file"]
    data = load_json_data(annotations_path)
    # flatten the annotations to each segment, df_annotations.index matches df_data.index
    yc2_annotations = data.annotations.apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame('annotations')
    yc2_annotations = yc2_annotations.annotations.apply(pd.Series)

    yc2_all = yc2_annotations.join(data, how='outer').reset_index()

    # create individual video_segment_id for each segment of a video
    yc2_all['video_seg_id'] = yc2_all['index'] + '_' + yc2_all['id'].apply(str)

    # match recipe_type with recipe_label
    yc2_all['recipe_label'] = yc2_all['recipe_type'].apply(int).map(recipe_labels_dic)
    yc2_all['recipe_index'] = yc2_all['recipe_label'].apply(lambda x: recipe_types.index(x))

    # split segments to single column
    yc2_all['seg_start'] = yc2_all['segment'].apply(lambda x: x[0])
    yc2_all['seg_end'] = yc2_all['segment'].apply(lambda x: x[1])

    yc2_all = yc2_all.drop(['level_0', 'segment'], axis=1)

    # re-split the dataset for the zero shot setting into a training, validation and test set
    # the splits are based on different recipe types
    yc2_all.recipe_type = yc2_all.recipe_type.apply(int)
    training = [205, 301, 309, 318, 121, 225, 207, 405, 314, 425, 324, 124, 422,
                223, 409, 323, 311, 214, 224, 226, 114, 310, 102, 103, 119, 112,
                222, 410, 111, 308, 127, 306, 419, 212, 208, 319, 108, 218, 206,
                423, 117, 115, 213, 110, 404, 304, 113, 325, 401, 317, 303, 203,
                302, 416, 230, 204, 406, 227, 221, 228, 421, 316, 116, 307, 109,
                219, 418, 413, 209, 106, 104, 321, 120, 105, 210]
    test = [201, 122, 313, 101, 229, 202, 412]
    validation = [215, 107, 216, 305, 403, 126, 211]


    yc2_all["subset_new"] = None
    yc2_all.loc[yc2_all['recipe_type'].isin(training), "subset_new"] = 'training'
    yc2_all.loc[yc2_all['recipe_type'].isin(validation), 'subset_new'] = 'validation'
    yc2_all.loc[yc2_all['recipe_type'].isin(test), 'subset_new'] = 'test'


    return yc2_all


def main(args):
    """
    Function that executes the pipeline of building the video features and index dataframe based on the given arguments
    :param args: arguments that specify from which source the frame-wise features should be built from
    :return:
    """

    data_dir = config.DATA["data_dir"]

    # create index data file with zero-shot split and store as csv file
    print("Create index dataframe")
    yc2_all = create_index_data_file(data_dir)

    yc2_all.to_csv(data_dir+config.DATA["data_all"], index=False)


    print("Build image features for the video segments")
    # load the frame-wise features for the video segments and save them as separate numpy files for each video segment
    if args.resNet:
        feature_dir_resnet = data_dir + "feat_csv/"
        save_dir_resnet = data_dir + "resnet_features/"

    if args.bn:
        feature_dir_bn = data_dir + "video_features"
        save_dir_bn = data_dir + "bn_features/"

    for idx in tqdm(range(len(yc2_all))):
        if args.resNet:
            get_resnet_feature(yc2_all, idx, feature_dir_resnet, save_dir_resnet)
        if args.bn:
            get_bn_feature(yc2_all, idx, feature_dir_bn, save_dir_bn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--resNet', dest='resNet', action='store_true')
    parser.add_argument('--bn', dest='bn', action='store_true')
    parser.set_defaults(resNet=False)
    parser.set_defaults(bn=False)
    args = parser.parse_args()
    main(args)
