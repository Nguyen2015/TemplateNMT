# -*- coding: utf-8 -*-

import glob
import json
import argparse
import re

import pandas as pd

from logic_util import parse_lambda
from syntactic_tree_parser import bpe_mask, un_bpe, un_bpe_mask

# recoverubg BPE words


def recover_bpe_words(row_info):
    if row_info["template"] == "error-parse-tree" or not isinstance(row_info["sentence"], str):
        row_info["bpe_template"] = 'error-parse-tree'
        row_info["status"] = "warning"
    else:
        natural_w = row_info["sentence"].split()
        bpe_w = row_info["bpe_sent"].split()
        bpe_template = row_info["template"]
        row_info["status"] = "normal"
        for i, w in enumerate(bpe_w):
            if "@@_" in w:
                if " {} ".format(natural_w[i]) not in bpe_template:
                    row_info["status"] = "warning"
                if w.replace("@@_", "") != natural_w[i]:
                    bpe_template = "err-pare-tree"
                    break
                if re.match(r'[^\w]', w[-1]):
                    bpe_template = bpe_template.replace(
                        " {} ".format(natural_w[i][:-1]), " {} ".format(w[:-1]), 1)
                else:
                    bpe_template = bpe_template.replace(
                        " {} ".format(natural_w[i]), " {} ".format(w), 1)
        row_info["bpe_template"] = bpe_template

    return row_info


def generate_template_mix(row_info, tag_freq=None):
    if row_info["bpe_template"] == "error-parse-tree":
        row_info["syntactic_template"] = row_info["bpe_sent"].replace("@@_", "@@ ") if isinstance(row_info["bpe_sent"], str) else ""
    else:
        bpe_template = parse_lambda(row_info["bpe_template"]) 
        top_frequent_pos = list(tag_freq.items())
        top_frequent_pos.sort(key=lambda x: x[1], reverse=True)
        top_frequent_pos = [x[0] for x in top_frequent_pos[:5]]

        bpe_template.flag_frequent_postag(top_frequent_pos)
        mix_template_nodes = bpe_template.scan_frequent_tree()

        row_info["syntactic_template"] = " ".join( [e.value.replace("@@_", "@@ ") for e in mix_template_nodes])
    return row_info

def generate_template(row_info, tag_freq=None):
    if row_info["bpe_template"] == "error-parse-tree":
        row_info["syntactic_template"] = row_info["bpe_sent"]
    else:
        if tag_freq is None:
            bpe_template = parse_lambda(row_info["bpe_template"])
            sent_len = len(row_info["bpe_sent"].split(" "))
            depth = min(max(sent_len*0.15, bpe_template.get_min_depth()),
                        bpe_template.get_max_depth())
            # max_depth = bpe_template.get_max_depth()
            # depth = min(max(max_depth - 2, bpe_template.get_min_depth()), max_depth)
            nodes_pruned = bpe_template.get_leaf_nodes_with_depth(depth)
            row_info["syntactic_template"] = " ".join(
                [e.value.replace("@@_", "@@ ") for e in nodes_pruned])
        else:
            bpe_template = parse_lambda(row_info["bpe_template"])
            min_depth = bpe_template.get_min_depth()
            max_depth = bpe_template.get_max_depth()
            if min_depth > max_depth: 
                print("----")
                print(row_info["bpe_sent"])
                print(row_info["bpe_template"])
                print(min_depth, max_depth)
                tmp_val = min(max_depth, min_depth)
                max_depth = tmp_val
                min_depth = tmp_val
            max_prob = -1
            max_nodes_pruned = None
            if min_depth == max_depth:
                max_nodes_pruned = bpe_template.get_leaf_nodes_with_depth(
                    min_depth)
            for d in range(min_depth, max_depth):
                nodes_pruned = bpe_template.get_leaf_nodes_with_depth(d)
                probs = []
                for n in nodes_pruned:
                    if n.value not in tag_freq:
                        continue
                    else:
                        probs.append(tag_freq[n.value])

                cur_prob = sum(probs) / len(probs) if len(probs) > 0 else 0
                if cur_prob > max_prob:
                    max_prob = cur_prob
                    max_nodes_pruned = nodes_pruned

            row_info["syntactic_template"] = " ".join(
                [e.value.replace("@@_", "@@ ") for e in max_nodes_pruned])
    return row_info

def generate_template_replace_np(row_info, tag_freq=None):
    if row_info["bpe_template"] == "error-parse-tree":
        row_info["syntactic_template"] = row_info["bpe_sent"]
    else:
        bpe_template = parse_lambda(row_info["bpe_template"])
        sent_len = len(row_info["bpe_sent"].split(" ")) 
        bpe_template.prune_tag(["NP"])
        nodes_pruned = bpe_template.get_leaf_nodes_template()
        row_info["syntactic_template"] = " ".join( [e.value.replace("@@_", "@@ ") for e in nodes_pruned])
    return row_info

def generate_template_replace_np_novp(row_info, tag_freq=None):
    if row_info["bpe_template"] == "error-parse-tree":
        row_info["syntactic_template"] = row_info["bpe_sent"]
    else:
        bpe_template = parse_lambda(row_info["bpe_template"]) 
        bpe_template.flag_vp_in_subtree()
        bpe_template.prune_tag_novp_in_subtree(["NP"])
        nodes_pruned = bpe_template.get_leaf_nodes_template()
        row_info["syntactic_template"] = " ".join( [e.value.replace("@@_", "@@ ") for e in nodes_pruned])
    return row_info

def generate_template_replace_npvp_nov(row_info, tag_freq=None):
    if row_info["bpe_template"] == "error-parse-tree":
        row_info["syntactic_template"] = row_info["bpe_sent"]
    else:
        bpe_template = parse_lambda(row_info["bpe_template"]) 
        bpe_template.flag_vp_in_subtree()
        bpe_template.prune_tag_novp_in_subtree(["NP", "VP"])
        nodes_pruned = bpe_template.get_leaf_nodes_template()
        row_info["syntactic_template"] = " ".join( [e.value.replace("@@_", "@@ ") for e in nodes_pruned])
    return row_info

def generate_template_replace_toptags_nov(row_info, tag_freq):
    if row_info["bpe_template"] == "error-parse-tree":
        row_info["syntactic_template"] = row_info["bpe_sent"]
    else:
        tag_top_freq = []
        tag_freq = list(tag_freq.items())
        tag_freq.sort(key=lambda x: x[1], reverse=True)
        tag_top_freq = [x[0] for x in tag_freq[:10]]
        
        bpe_template = parse_lambda(row_info["bpe_template"]) 
        bpe_template.flag_vp_in_subtree()
        bpe_template.prune_tag_novp_in_subtree(tag_top_freq)
        nodes_pruned = bpe_template.get_leaf_nodes_template()
        
        row_info["syntactic_template"] = " ".join( [e.value.replace("@@_", "@@ ") for e in nodes_pruned])
    return row_info

def generate_template_depth3(row_info, tag_freq, depth_level=2):
    if row_info["bpe_template"] == "error-parse-tree":
        row_info["syntactic_template"] = "S"
    else:
        bpe_template = parse_lambda(row_info["bpe_template"])
        max_nodes_pruned = bpe_template.get_leaf_nodes_with_depth(depth_level)

        row_info["syntactic_template"] = " ".join(
            [e.value.replace("@@_", "@@ ") for e in max_nodes_pruned])
    return row_info


def bpe_template_aggregate(base_path="./data/iwslt14.tokenized.de-en/"):
    for file_name in glob.glob(base_path + "*.template"):
        print(file_name)
        file_name_org = file_name.replace(".template", ".bpe")
        with open(file_name_org, "rt", encoding="utf8") as f:
            lines = [_l.strip().replace("@@ ", "@@_") for _l in f.readlines()]
        df_org = pd.DataFrame({'bpe_sent': lines})
        df_syntactic = pd.read_csv(file_name)
        df = pd.concat(
            [df_org, df_syntactic[["sentence", "template"]]], axis=1, sort=False)
        df = df.apply(recover_bpe_words, axis=1, result_type='expand')
        df[["bpe_sent", "template", "status", "bpe_template"]].to_csv(
            file_name.replace(".template", ".bpetemplate"))


def save_syntactic_template_file(base_path="./data/iwslt14.tokenized.de-en/"):
    for file_name in glob.glob(base_path + "*.syntactictemplate"):
        print(file_name)
        df = pd.read_csv(file_name)

        # df[["syntactic_template"]].to_csv(file_name.replace(".syntactictemplate", ".softtemplate"),
        #                                   index=False,
        #                                   header=False,
        #                                   sep="€"
        #               
        lines = df["syntactic_template"].values.tolist()
        with open(file_name.replace(".syntactictemplate", ".softtemplate"), "wt", encoding="utf8") as f:
            lines = [l if isinstance(l, str) else "S" for l in lines]
            f.write("\n".join(lines))


def syntactic_template_aggregate(base_path="./data/iwslt14.tokenized.de-en/", use_tag_freq=False, generate_template_func=None, **kwargs):
    if use_tag_freq:
        print("generate tag - freq")
        tag_freq = {}
        for file_name in glob.glob(base_path + "*.bpetemplate"):
            if "train" in file_name:
                tag_freq = json.load(open(file_name.replace(
                    ".bpetemplate", ".tagfreq.json"), "rt", encoding='utf8'))
    else:
        tag_freq = None

    for file_name in glob.glob(base_path + "*.bpetemplate"):
        print(file_name)
        df = pd.read_csv(file_name)
        df = df.apply(generate_template_func, axis=1,
                      result_type='expand', tag_freq=tag_freq, **kwargs)
        df[['bpe_sent', "syntactic_template"]].to_csv(
            file_name.replace(".bpetemplate", ".syntactictemplate"))


def extract_postag(row_info):
    node_names = []
    if row_info["bpe_template"] == "error-parse-tree":
        pass
    else:
        bpe_template = parse_lambda(row_info["bpe_template"])
        node_names = bpe_template.get_all_node_name()
    row_info["tags"] = node_names
    return row_info


def stats_postags(base_path="./data/iwslt14.tokenized.de-en/"):
    for file_name in glob.glob(base_path + "*.bpetemplate"):
        print(file_name)
        if "train" not in file_name:
            continue
        df = pd.read_csv(file_name)
        df = df.apply(extract_postag, axis=1, result_type='expand')
        tag_freq = {}
        for sent_tags in df['tags'].values:
            for tag in sent_tags:
                if tag not in tag_freq:
                    tag_freq[tag] = 1
                else:
                    tag_freq[tag] += 1

        total_tag = sum(tag_freq.values()) + 0.0
        for k, v in tag_freq.items():
            tag_freq[k] = v / total_tag

        json.dump(tag_freq, open(file_name.replace(
            ".bpetemplate", ".tagfreq.json"), "wt", encoding="utf8"))
        print(tag_freq)
        return tag_freq


def srl_template_parse_(sentence, srl_predictor_bert, return_w_align=False):
    sentence_unbpe = un_bpe(sentence)
    w_align = [(bpe_mask(w1), w2) for w1, w2 in zip(un_bpe_mask(
        sentence).split(" "), sentence_unbpe.split(" ")) if w1 != w2]
    srl_s = srl_predictor_bert.predict(sentence_unbpe)
    if len(srl_s['verbs']) == 0:
        if return_w_align:
            return sentence, []
        else:
            return sentence

    count_tags = [len([tag for tag in x['tags'] if tag != 'O'])
                  for x in srl_s['verbs']]

    # get max idx
    max_v = -1
    idx_selection = 0
    for i, v in enumerate(count_tags):
        if v > max_v:
            idx_selection = i
            max_v = v

    # get value in max idx
    srl_des = srl_s['verbs'][idx_selection]['description']

    if return_w_align:
        return srl_des, w_align
    else:
        return srl_des


def srl_template_parse(sentence, srl_predictor_bert):
    srl_des, w_align = srl_template_parse_(
        sentence, srl_predictor_bert, return_w_align=True)

    # replace sentence to template
    srl_des = re.sub(r'\[V\: ([^\]]*)\]', r'\1', srl_des)
    srl_des = re.sub(r'\[([^\]]*)\: [^\]]*\]', r'\1', srl_des)

    # recover bpe for some words
    template_w = srl_des.split(" ")
    for i in range(len(template_w)):
        w = template_w[i]
        for w1, w2 in w_align:
            template_w[i] = w1 if w2 == w else w

    return " ".join(template_w)


def generate_srl_template(file_pattern="../lang2logic_parser/data-sem/iwslt142/Y_*_5.tsv"):
    from allennlp_models.pretrained import load_predictor

    srl_predictor_bert = load_predictor('structured-prediction-srl-bert')

    for file_name in glob.glob(file_pattern):
        print(file_name)
        with open(file_name, "rt", encoding="utf8") as _f:
            df = pd.DataFrame(
                data={'sentence': [_l.strip() for _l in _f.readlines()]})
            # df['srl'] = df['sentence'].apply(
            #     srl_template_parse_, srl_predictor_bert=srl_predictor_bert)
            df['template'] = df['sentence'].apply(
                srl_template_parse, srl_predictor_bert=srl_predictor_bert)
            print(df.head())

            # df.to_excel( file_name + ".xlsx")
            df[['template']].to_csv(
                file_name + ".template", index=False, sep='£', header=False)


def replace_punct_fn(sentence):

    # replace sentence to template
    sentence = re.sub(r' [\,\.\:\;\$\#\!\?\@]( |$)', ' PUNCT ', sentence)
    return sentence.strip()


def replace_punct_token(file_pattern="../lang2logic_parser/data-sem/iwslt142/Y_*_5.tsv"):

    for file_name in glob.glob(file_pattern):
        print(file_name)
        with open(file_name, "rt", encoding="utf8") as _f:
            df = pd.DataFrame(
                data={'templ': [_l.strip() for _l in _f.readlines()]})

            df['templ_punct'] = df['templ'].apply(replace_punct_fn)
            print(df.head())

            # df.to_excel( file_name + ".xlsx")
            df[['templ_punct']].to_csv(
                file_name + ".replace_punct.tsv", index=False, sep='£', header=False)


def concat_src_templ(row_info):
    return '{} [SEP] {}'.format(row_info['src'], row_info['templ'])


def concat_template_with_source(file_src, file_templ, file_out):

    with open(file_src, "rt", encoding="utf8") as _f:
        src_lines = [_l.strip() for _l in _f.readlines()]
    with open(file_templ, "rt", encoding="utf8") as _f:
        templ_lines = [_l.strip() for _l in _f.readlines()]

    df = pd.DataFrame(
        data={'src': src_lines, 'templ': templ_lines})

    df['src_templ'] = df[['src', 'templ']].apply(concat_src_templ, axis=1)
    print(df.head())

    # df.to_excel( file_name + ".xlsx")
    df[['src_templ']].to_csv(file_out, index=False, sep='£', header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_folder_or_pattern',
                        action="store", dest="path_folder_or_pattern",
                        help="path folder saving data or pattern matching file data", default='path/to/folder_or_pattern')
    parser.add_argument('--type_template',
                        action="store", dest="type_template",
                        help="type of the tempalte: syntactic | srl", default='nouse')
    parser.add_argument('--depth_level',
                        action="store", dest="depth_level", type=int,
                        help="depth_level to generate template", default=2)
    parser.add_argument('--mix_pos_word',
                        action="store_true", dest="mix_pos_word",
                        help="mix postag with words", default=False)
    parser.add_argument('--use_dyn_templ',
                        action="store_false", dest="use_dyn_templ",
                        help="using dynamic template or not", default=True)
    parser.add_argument('--replace_punct',
                        action="store_true", dest="replace_punct",
                        help="replace some punct token into PUNCT", default=False)
    parser.add_argument('--concat_template',
                        action="extend", dest="concat_template", type=str, nargs="*",
                        help="concat template with source sentence", default=[])
    options = parser.parse_args()

    if options.type_template == 'srl':
        generate_srl_template(file_pattern=options.path_folder_or_pattern)
    elif options.type_template == 'syntactic':
        bpe_template_aggregate(base_path=options.path_folder_or_pattern)
        if options.use_dyn_templ:
            print("stats postag")
            stats_postags(base_path=options.path_folder_or_pattern)
        if options.mix_pos_word:
            syntactic_template_aggregate(
                base_path=options.path_folder_or_pattern, use_tag_freq=True, generate_template_func=generate_template_mix)
        else:
            print("gen template")
            syntactic_template_aggregate(
                base_path=options.path_folder_or_pattern, use_tag_freq=options.use_dyn_templ, generate_template_func=generate_template)
        save_syntactic_template_file(base_path=options.path_folder_or_pattern)
    elif options.type_template == 'np':
        bpe_template_aggregate(base_path=options.path_folder_or_pattern)
        syntactic_template_aggregate(
            base_path=options.path_folder_or_pattern, generate_template_func=generate_template_replace_np)
        save_syntactic_template_file(base_path=options.path_folder_or_pattern)
    elif options.type_template == 'np_nov':
        syntactic_template_aggregate(
            base_path=options.path_folder_or_pattern, generate_template_func=generate_template_replace_np_novp)
        save_syntactic_template_file(base_path=options.path_folder_or_pattern)
    elif options.type_template == 'npvp_nov':
        syntactic_template_aggregate(
            base_path=options.path_folder_or_pattern, generate_template_func=generate_template_replace_npvp_nov)
        save_syntactic_template_file(base_path=options.path_folder_or_pattern)
    elif options.type_template == 'toptags_nov':
        stats_postags(base_path=options.path_folder_or_pattern)
        syntactic_template_aggregate(
            base_path=options.path_folder_or_pattern, generate_template_func=generate_template_replace_toptags_nov, 
            use_tag_freq=True)
        save_syntactic_template_file(base_path=options.path_folder_or_pattern)
    elif options.type_template == 'd3':
        bpe_template_aggregate(base_path=options.path_folder_or_pattern)
        syntactic_template_aggregate(
            base_path=options.path_folder_or_pattern, generate_template_func=generate_template_depth3, 
            use_tag_freq=False, depth_level=options.depth_level)
        save_syntactic_template_file(base_path=options.path_folder_or_pattern)

    if options.replace_punct:
        replace_punct_token(file_pattern=options.path_folder_or_pattern)

    if len(options.concat_template) > 0:
        concat_template_with_source(
            file_src=options.concat_template[0], file_templ=options.concat_template[1], file_out=options.concat_template[0]+".srctempl.tsv")
