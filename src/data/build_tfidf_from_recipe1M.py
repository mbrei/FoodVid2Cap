import pandas as pd
import numpy as np
import sys
import argparse
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

sys.path.append('C:/Users/user/foodcap')

# misc
import src.config as config
from src.utils import save_list_to_file

def create_list(x):
    '''
    Creates a list of strings removes unneccessary characters from a string.

    Input:
    :param x: string, that should be processed

    Output:
    :return: cleaned string
    '''
    result = []
    for dic in x:
        text = dic['text']
        text = text.replace('.', '')
        text = text.replace(';', '')
        text = text.replace(',', '')
        text = text.replace('(', '')
        text = text.replace(')', '')
        text = text.replace('*', '')
        text = text.replace('!', '')
        text = text.replace(':', '')
        text = text.replace('/', '')
        text = text.replace('[', '')
        text = text.replace(']', '')
        text = text.replace('+', '')
        text = text.replace('%', '')
        text = text.lower()
        result.append(text)
    return result


def get_all_instruction_step(df, recipe_type):
    '''
    Concat all instructions of a recipe type and append it to one document.

    :param df: pandas dataframe, that stores the instruction
    :param recipe_type: string, the to consideres recipe type
    :return: list, consisting of all instructions
    '''
    df_rt = df[df['recipe_type'] == recipe_type]

    document = []
    il = list(df_rt.instructions_list.values)
    for steps in il:
        document.extend(steps)
    return document

def main(args):
    # read in the annotations, instructions step of Recipe1M and different recipe_types (external ressource)
    df = pd.read_json(config.DATA["data_dir"]+config.DATA["layers_file"])
    df_recipe_titles = pd.read_csv(config.DATA["data_dir"]+config.DATA["recipe_titles_yc2"], header=None)
    df_recipe_titles = df_recipe_titles.reset_index()
    df_recipe_titles.columns = ['recipe_index', 'recipe_type', 'recipe_label']

    df['title'] = df['title'].str.lower()
    df_recipe_titles['recipe_label'] = df_recipe_titles['recipe_label'].str.lower()

    # Find recipe instruction from external resource Recipe1M that matches recipe types in YouCook2 dataset
    # Dictionary for correcting misspelled recipe labels, or similar recipe types
    correction_dic = {
        "pizza marghetta": ["pizza margherita"],
        "salmon nigiri": [" nigiri"],
        "vietnam spring roll": ["spring roll"],
        "porkolt hungarian stew": ["porkolt","hungarian beef and onion stew"],
        "yaki udon noodle": ["yaki udon"],
        "authentic japanese ramen": ["ramen"],
        "vietnam sandwish": ["sandwich"],
        "roti jala": ["roti"], 
        "wanton noodle": ["wanton noodle"], 
        "singapore curry laksa": ["curry"],
        "spicy tuna roll":["tuna roll"],
        "galbi":["korean ribs"],
        "dal makhani":["dal makhni"],
        "currywurst":["curry sausage"],
        "indian lamb curry":["lamb curry"],
        "spider roll":["maki sushi"],
        "thai red curry chicken":["thai red curry"],
        "sichuan boiled fish":["sichuan fish"],
        "singapore rice noodle":["singapore noodle"]
    }
    
    df_list = []
    for recipe_label in df_recipe_titles['recipe_label']:
        print('Search for', recipe_label, 'in Recipe1Million cooking instructions.')
        df_rl = df[df['title'].str.contains(recipe_label) == True]
        df_rl['recipe_type'] = recipe_label
        df_list.append(df_rl)
    
    for recipe_label in correction_dic.keys():
        for cor_recipe_label in correction_dic[recipe_label]:
            print('Search for', recipe_label, 'as', cor_recipe_label, 'in Recipe1Million cooking instructions.')
            df_rl = df[df['title'].str.contains(cor_recipe_label) == True]
            df_rl['recipe_type'] = recipe_label
            df_list.append(df_rl)
    
    # Concat all matching instructions of the recipe types
    df_r1m = pd.concat(df_list)

    # Clean all instructions
    df_r1m['instructions_list'] = df_r1m.instructions.apply(create_list)

    print('Concat all found instruction steps of Recipe1Million.')
    recipe_list = []
    # Loop through all recipe_labels, collect all instructions of every corresponding recipe
    for i in tqdm(range(len(df_recipe_titles))):
        recipe_type = df_recipe_titles.loc[i]['recipe_label']
        doc = get_all_instruction_step(df_r1m, recipe_type)
        doc_all = ' '.join(doc)
        recipe_list.append(doc_all)

    # Calculate tfidf-vector for entire list of all recipe types
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(recipe_list)

    # Save all occured words of all instructions and calculated weights
    save_list_to_file(vectorizer.get_feature_names(),config.DATA["data_dir"], 'R1M_vocab.json')
    save_npz(config.DATA["data_dir"]+'tfidf_weight_matrix.npz', X)
    df_recipe_titles.to_csv(config.DATA["data_dir"]+'label_index_foodtype.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
