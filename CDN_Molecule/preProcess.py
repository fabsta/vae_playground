import argparse
import pandas as pd
import numpy as np
import json
import pickle
import sys
from io import StringIO


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def get_smile2drug():
    char2num = load_json_file('data/char2num.json')
    drug2idx = {}
    idx2drug = {}
    smile2drug = {}
    data = []
    idx = 0
    for i in open('data/FDA_drugs.smi'):
        try:
            smile, name = i.split(' ')[:2]
        except:
            print(i)
            continue
        reject = False
        for char in list(smile):
            if char not in char2num:
                reject = True
        if len(smile) > (max_length - 2) or reject:
            continue
        if name.lower() in list(drug2idx.keys()):
            continue
        idx2drug[idx] = name.lower()
        drug2idx[name.lower()] = idx
        smile2drug[smile] = name
        data.append(smile)
        idx += 1

    return smile2drug


def mol_analysis(smile, real=None):
    sys.stderr = StringIO()
    from rdkit import Chem
    m = Chem.MolFromSmiles(smile)
    if m is not None:
        print("Such molecule exist:")
        print(smile)
        if real is not None:
            print("Real")
            print(real)
            print("same? " + str(smile == real))
            print("levenshtein distance: " + str(levenshtein_distance(real, smile)))
            print("")
        return 1
    return 0


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def write_to_json(path, obj):
    f = open(path, 'w')
    json.dump(obj, f)
    f.close()
    return 0


def load_json_file(path):
    with open(path) as data_file:
        data = json.load(data_file)
    return data


def smiles_2_num_vec(smile, char2num):
    vec = np.zeros(max_length, dtype=np.int32)
    chars = list(smile)
    counter = 1
    vec[0] = GO
    """
    for char in chars:
        vec[counter] = char2num[char]
        counter += 1"""
    for id, char in enumerate(chars):
        try:
            vec[counter] = char2num[char]
            counter += 1
        except:
            print("Error with {} {}".format(id, char))

    vec[counter] = END_OF_MOL
    counter += 1
    for i in range(max_length-counter):
        vec[counter] = PAD
        counter += 1
    return vec


def create_dictionaries(chars_dict):
    char2num = {}
    num2char = {}
    for num, char in enumerate(chars_dict.keys()):
        num2char[num] = char
        char2num[char] = num

    write_to_json('data/char2num.json', char2num)
    write_to_json('data/num2char.json', num2char)
    return num2char, char2num


def chars_freq(smiles):
    dictionary = {}
    for index, smile in enumerate(smiles):
        if index % 1000 == 0: print(index)
        temp = list(smile)
        for char in temp:
            if char in list(dictionary.keys()):
                dictionary[char] += 1
            else:
                dictionary[char] = 1
    return dictionary


def smiles_length_dist(smiles):
    lengths = {}
    for smile in smiles:
        length = int(len(smile))
        if length in lengths:
            lengths[length] += 1
        else:
            lengths[length] = 1
    write_to_json('data/smiles_lengths_dist.json', lengths)


def create_zinc_2_vectors(smiles, char2num):
    smiles_data = []
    print("Num of input molecules: ", str(len(smiles)))
    for index, smile in enumerate(smiles):
        print(index)
        if len(smile) > (max_length - 2):
            continue
        smiles_data.append(smiles_2_num_vec(smile, char2num))

    print("Num of output molecules: ", str(len(smiles_data)))
    pickle.dump(smiles_data, open(output_file, 'wb'))
    return np.array(smiles_data)


def process_data():
    # Load drug-like data
    data = np.squeeze(pd.read_csv(input_file).values)

    if write_json > 0:
        # Get char frequencies
        chars_dict = chars_freq(data)
        # Create dictionaries
        num2char, char2num = create_dictionaries(chars_dict)
    else:
        char2num = load_json_file('data/char2num.json')

    if smiles_dist > 0:
        smiles_length_dist(data)
    # Create train vectors
    if return_output > 0:
        create_zinc_2_vectors(data, char2num)
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing.')
    parser.add_argument('--max_length', '-m', type=int,
                    default=50, help='maximum length of smiles')
    parser.add_argument('--vocabulary_size', '-v', type=int,
                    default=34, help='vocabulary size')
    parser.add_argument('--input_file', '-i', type=str,
                    default='data/250k_rndm_zinc_drugs_clean.smi',
                    help="Path to input SMILES file")
    parser.add_argument('--output_file', '-o', type=str,
                    default='data/TrainVectors.pickle',
                    help="Path to output pickle file")
    parser.add_argument('--write_json', '-j', type=int,
                        default=0, help="Write json dictionary files if 1")
    parser.add_argument('--smiles_dist', '-d', type=int,
                        default=0, help="Write smiles length distribution as json if 1")
    parser.add_argument('--return_output', '-r', type=int,
                        default=1, help="Return output pickle file if 1")
    args = parser.parse_args()

    max_length = args.max_length
    vocabulary_size = args.vocabulary_size
    GO = vocabulary_size
    END_OF_MOL = vocabulary_size + 1
    PAD = vocabulary_size + 2
    input_file = args.input_file
    output_file = args.output_file
    write_json = args.write_json
    smiles_dist = args.smiles_dist
    return_output = args.return_output

    process_data()



