

import argparse
import glob

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_folder',
                        action="store", dest="path_folder", type=str,
                        help="path folder saving data or pattern matching file data", default='path/to/folder_or_pattern')

    parser.add_argument('--method',
                        action="store", dest="method", type=str,
                        help="method to run: rename_template", default='rename_template')

    options = parser.parse_args()
    if options.method == "rename_template":
        for file in glob.glob(options.path_folder + "*.softtemplate"):

            print(file)
            base_name = os.path.basename(file)
            data_dir = os.path.dirname(file)
            file_name = base_name.split(".")[0]
            os.rename(file, data_dir + "/" + file_name + ".tsv")
