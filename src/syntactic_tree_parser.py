import glob
import os
import re
import pandas as pd
import spacy
from benepar.spacy_plugin import BeneparComponent
import argparse
import os.path

import nltk
import benepar
benepar.download('benepar_en3')
from stanfordcorenlp import StanfordCoreNLP

nlp = None # StanfordCoreNLP(path_or_host='../semparser/stanford-corenlp-4.0.0/', lang='de')


def un_bpe(sentence):
    return sentence.replace("@@ ", "")

def bpe_mask(sentence):
    return sentence.replace("@@_", "@@ ")

def un_bpe_mask(sentence):
    return sentence.replace("@@ ", "@@_")

def sentence_feature_extractor(sentence, nlp=None):
    nlp = nlp if nlp is not None else StanfordCoreNLP(path_or_host='../semparser/stanford-corenlp-4.0.0/', lang='de')
    
    try:
        r_dict = nlp._request('pos,parse', sentence)
        doc = " ".join([s['parse'] for s in r_dict['sentences']])
        # doc = nlp.parse(sentence)
    except Exception as e:
        print("[Except] parse sentence {}".format(sentence))
        print(e)
        doc = "error-parse-tree"
    doc = doc.replace('\n', ' ')
    doc = re.sub(r'([\(\)])', r' \1 ', doc)
    return re.sub(r' {2,}', ' ', doc.strip())


# nlp = spacy.load('vi_spacy_model')
# nlp.add_pipe(BeneparComponent('benepar_en'))


def envi_sentence_feature_extractor(sentence):
    try:
        doc = nlp(sentence)
        doc = " ".join([s._.parse_string for s in list(doc.sents)])
    except Exception as e:
        print("[Except] parse sentence {}".format(sentence))
        print(e)
        doc = "error-parse-tree"
    doc = doc.replace('\n', ' ')
    doc = re.sub(r'([\(\)])', r' \1 ', doc)
    return re.sub(r' {2,}', ' ', doc.strip())


def _normalize_row(row_info):
    natural_sentence = row_info["sentence"]
    template = row_info["template"]
    for e in list(re.findall(r'\([^\(\)]*\)', template)):
        w_check = e.split(" ")[2]
        if "_" in w_check and w_check not in natural_sentence and w_check.replace("_", " ") in natural_sentence:
            new_e = e.replace(w_check, w_check.replace("_", " "))
            template = template.replace(e, new_e)

    row_info["template"] = template
    return row_info


def normalize_token(path_file_df):
    df_syntactic = pd.read_csv(path_file_df)
    df_syntactic = df_syntactic.apply(_normalize_row, axis=1, result_type='expand')
    df_syntactic.to_csv(path_file_df)
    return df_syntactic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_folder_or_pattern',
                      action="store", dest="path_folder_or_pattern", type=str,
                      help="path folder saving data or pattern matching file data", default='path/to/folder_or_pattern')

    parser.add_argument('--path_stanford_lib',
                      action="store", dest="path_stanford_lib",
                      help="path folder saving stanford library", default='path/to/stanford_lib')
    parser.add_argument('--lang',
                      action="store", dest="lang",
                      help="language: en|de", default='en')
    parser.add_argument('--port',
                        action="store", dest="port", type=int,
                        help="port stanford service", default=None)

    options = parser.parse_args()
    base_path = os.path.dirname(options.path_folder_or_pattern)
    
    path_folder_or_pattern = options.path_folder_or_pattern.replace(" x£x", "")
    for file_name in glob.glob(path_folder_or_pattern):
        print(file_name)

        if os.path.isfile(file_name + ".template"):
            print("[Skip] this file {}, because it already parsed".format(file_name))
            continue

        with open(file_name, "rt", encoding="utf8") as _f:
            new_data = []
            df = pd.DataFrame(data={'sentence': [un_bpe(bpe_mask(_l.strip())) for _l in _f.readlines()]})
            with StanfordCoreNLP(path_or_host=options.path_stanford_lib, lang=options.lang, memory='8g', port=options.port) as nlp_obj:
                df['template'] = df['sentence'].apply(sentence_feature_extractor, nlp=nlp_obj)
                print(df.head())
                df.to_csv(file_name + ".template")
                nlp_obj.close()

        # for file_name in glob.glob(base_path + "*.template"):
        print('normalize', file_name + ".template")
        normalize_token(path_file_df=(file_name + ".template"))