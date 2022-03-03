import glob

import pandas as pd


def count_intersect_words(row_info):
    bpe_sent = row_info["bpe_sent"].replace("@@_", "@@ ").split(" ")
    syntactic_template = row_info["syntactic_template"].replace("@@_", "@@ ").split(" ")
    row_info["count_intersect_w"] = len([w for w in bpe_sent if w in syntactic_template])
    row_info["count_template_w"] = len(syntactic_template)
    return row_info


def stats_intersect_words():
    base_path = "./data/iwslt14.tokenized.de-en/"
    for file_name in glob.glob(base_path + "*.syntactictemplate"):
        print(file_name)
        df = pd.read_csv(file_name)
        df = df.apply(count_intersect_words, axis=1, result_type='expand')
        count_intersect_w = sum(df["count_intersect_w"].values)
        count_template_w = sum(df["count_template_w"].values)
        print(count_intersect_w / count_template_w)


def count_tags(row_info):
    bpe_sent = row_info["bpe_sent"].replace("@@_", "@@ ").split(" ")
    syntactic_template = row_info["syntactic_template"].replace("@@_", "@@ ").split(" ")
    row_info["count_intersect_w"] = len([w for w in bpe_sent if w in syntactic_template])
    row_info["count_template_w"] = len(syntactic_template)
    return row_info


def stats_tags_in_syntactic_tree():
    base_path = "./data/iwslt14.tokenized.de-en/"
    for file_name in glob.glob(base_path + "*.syntactictemplate"):
        print(file_name)
        df = pd.read_csv(file_name)
        df = df.apply(count_intersect_words, axis=1, result_type='expand')
        count_intersect_w = sum(df["count_intersect_w"].values)
        count_template_w = sum(df["count_template_w"].values)
        print(count_intersect_w / count_template_w)


if __name__ == "__main__":
    stats_tags_in_syntactic_tree()
