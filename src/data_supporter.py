import glob
import json
import pandas as pd
import argparse
import re


def add_number_to_row(file_in="path/to/file", file_out="path/to/file"):

    with open(file_in, 'rt') as f:
        lines = [l.strip() for i, l in enumerate(f.readlines())]
        lines_number = [i for i in range(len(lines))]
    df = pd.DataFrame(data={"rid": lines_number, "sentence": lines})
    df.to_csv(file_out, index=False)


template_cleaner = re.compile(r"\s{2,}")


def template_clean(template: str):
    template = template_cleaner.sub(" ", template.replace(
        "(", " ( ").replace(")", " ) ").strip())
    return template


def add_segment_emb(template: str):
    template = template_cleaner.sub(" ", template.replace(
        "(", " ( ").replace(")", " ) ").strip())
    return template


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_folder_or_pattern',
                        action="store", dest="path_folder_or_pattern",
                        help="path folder saving data or pattern matching file data", default='path/to/folder_or_pattern')
    parser.add_argument('--action',
                        action="store", dest="action", type=str,
                        help="action run: split | mergesubfile | addnumber | addsegmentemb", default="")
    parser.add_argument('--splitted_size',
                        action="store", dest="splitted_size", type=int,
                        help="size of splited file", default=1000)
    parser.add_argument('--alignment_file:',
                        action="store", dest="alignment_file", type=str,
                        help="path file alignment, natural sentence", default='path/to/file')

    options = parser.parse_args()
    if options.action == "split":
        for file_name in glob.glob(options.path_folder_or_pattern):
            with open(file_name, "rt", encoding="utf8") as f:
                lines = [_l.strip() for _l in f.readlines()]

            len_file = len(lines)
            count_sub_file = len_file // options.splitted_size + \
                (1 if len_file % options.splitted_size > 0 else 0)
            meta_data = [{"file_name": "{}.sub{}.tsv".format(file_name, i),
                          "index": [i*options.splitted_size, min((i+1)*options.splitted_size, len_file)]
                          } for i in range(count_sub_file)]
            json.dump(meta_data, open("{}.meta.json".format(
                file_name), "wt", encoding="utf8"), indent=1)

            for info in meta_data:
                with open(info["file_name"], "wt", encoding="utf8") as f:
                    f.write(
                        "\n".join(lines[info["index"][0]: info["index"][1]]))
            print(meta_data)

    if options.action == "mergesubfile":
        for file_name in glob.glob(options.path_folder_or_pattern.strip()):
            meta_info = json.load(open(file_name, "rt"))
            file_out = file_name.replace(".meta.json", ".template")
            df = pd.DataFrame()
            for sub_file in meta_info:
                print(sub_file["file_name"])
                df = df.append(pd.read_csv(
                    sub_file["file_name"] + ".template"))
            df[["sentence", "template"]].to_csv(file_out)
            print("Writing file {}, len = {}".format(file_out, len(df)))

    if options.action == "merge":
        with open(options.alignment_file, 'rt') as f:
            sents = [l.strip().replace("@@ ", "") for l in f.readlines()]
        templates = pd.DataFrame()
        for file_name in glob.glob(options.path_folder_or_pattern.strip()):
            sub_data = pd.read_csv(
                open(file_name), index_col=0, escapechar='\\', names=["rid", "template"])

            templates = templates.append(sub_data)

        if len(templates) == len(sents) and len(sents) > 0:
            templates = templates.sort_index()
            templates['sentence'] = sents
            templates['template'] = templates['template'].apply(template_clean)
            file_out = options.alignment_file + ".template"
            print("Writing file ... {}".format(file_out))
            templates[['sentence', 'template']].to_csv(
                file_out, index=True, header=True)
        else:
            print("Error count len: src={}, templ={}".format(
                len(sents), len(templates)))

    if options.action == "add-row-id":
        for file_name in glob.glob(options.path_folder_or_pattern.strip()):
            add_number_to_row(file_name, file_name + ".line.csv")

    if options.action == "addsegmentemb":
        seg_ment_list = list("XT")
        for file_name in glob.glob(options.path_folder_or_pattern.strip()):
            new_data_lines = []
            with open(file_name) as f:
                for l in f.readlines():
                    data = l.strip().split("[SEP]")
                    new_line = []
                    for i, seg_v in enumerate(data):
                        new_seg_v = []
                        segment_name = seg_ment_list[i]
                        for w in seg_v.strip().split(" "):
                            new_w = "{}￨{}".format(w, segment_name)
                            new_seg_v.append(new_w)
                        new_line.append(" ".join(new_seg_v))
                        new_line.append("[SEP]￨{}".format(segment_name))
                    if len(new_line) > 1:
                        new_line = new_line[:-1]
                    new_data_lines.append(" ".join(new_line))
            file_out = file_name+".segmentid.tsv"
            print("Writing file .. {}".format(file_out))
            with open(file_out, "wt", encoding="utf8") as f:
                f.write("\n".join(new_data_lines))
