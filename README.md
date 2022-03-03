# Multi-sources Transformer for Neural Machine Translation 
## 1. Folder structure information

The files in path `data/iwslt14deen` can be coppied from our provided Supplementary Material data for skipping time-consuming preprocess step. 
```
./release_code/
├── data
│   └── iwslt14deen         # save files language pairs and template for each datasets.
│       ├── test.de     
│       ├── test.en    
│       ├── train.de    
│       ├── train.en    
│       ├── valid.de    
│       ├── valid.en   
|       └── ....
├── nmt_py38_env/           # python virtual environment 
│       └── stanford-corenlp-4.0.0/ # the stanford NLP tool for parsing syntactic tree 
├── README.md
├── requirements.txt        # requirement python packages
├── scripts                 # contains all script to run 
│   ├── run1.transformer.sh
│   └── ...
└── src                     # CODE for generate template and model
    ├── fairseqSyntaxNMT # CODE for Transformer Multi-sources Encoder(TME) model 
    │   ├── __init__.py
    │   ├── multisources_transformer_model_bak.py
    │   ├── multisources_transformer_model.py
    │   ├── template_language_pair_dataset.py
    │   └── template_translation_task.py
    ├── data_supporter.py
    ├── eval_metrics.py
    ├── logic_util.py
    ├── prepare_data_supporter.py
    ├── syntactic_tree_parser.py
    ├── template_generator.py
    └── verify_data.py
```

## 2. Prepare data and environments

- build virtual env python with name: nmt_py38_env/ and install all package in the `requirements.txt`: 
  
  ```bash 
  conda  create --prefix nmt_py38_env python=3.8 
  source activate nmt_py38_env/
  pip install -r ./requirements.txt
  ``` 
- download the stanford nlp library [link](http://nlp.stanford.edu/software/stanford-corenlp-4.0.0.zip) and all necessary corresponding models [link](http://nlp.stanford.edu/software/stanford-corenlp-4.0.0-models-german.jar) (e.g. [german](http://nlp.stanford.edu/software/stanford-corenlp-4.0.0-models-german.jar), french, ...)
- unzip the standford as name in folder structure (if you download stanford-lib new version, please rename to the same in folder structure information `./nmt_py38_env/stanford-corenlp-4.0.0/`)
  ```bash  
  wget http://nlp.stanford.edu/software/stanford-corenlp-latest.zip (http://nlp.stanford.edu/software/stanford-corenlp-4.0.0.zip)&& unzip stanford-corenlp-latest.zip
  mv stanford-corenlp-4.0.0 nmt_py38_env/stanford-corenlp-4.0.0 # note: the stanford corenlp library name maybe different in template  stanford-corenlp-x.x.x, therefore, pls move it to the envs directory in path "nmt_py38_env/stanford-corenlp-4.0.0" for correct path in our scripts 
  ```
- download, preprocess and unzip the necessary data `./data/iwslt14deen`. Note: this code need 6 files above for each dataset.
  ```bash  
  source activate nmt_py38_env/
  bash ./scripts/iwslt14deen_data_prepare.sh
  ls data/iwslt14deen
  # data/iwslt14deen:
  # code  test.de  test.en  train.de  train.en  valid.de  valid.en
  ```

## 3. Run

All steps in Section `2. Prepare data and environments` need to sucessfully finish.

### 3.1. Run baseline model 
- from root data `./release_code/`, run each scripts file for each run that you want. 
    Example:
    ```bash
    bash scripts/run1.transformer.baseline.sh
    ```

    > **Note**: this script is designed for **IWSLT2014 de-en** dataset, training process take 7 hours to complete on 1 GPU Tesla P100-PCIE. </br>The output: 
    BLEU = 35.93, 69.7/44.4/30.4/21.1 (BP=0.957, ratio=0.958, hyp_len=125668, ref_len=131156)

### 3.2. Run TME +DepT / TME +ProbT / TME +LenT settings in Source side and target side 

- generate template data  + training model on Source side 
  ```bash
  bash ./scripts/run10.tme.dept.sh
  # bash ./scripts/run9.tme.probt.sh
  # bash ./scripts/run8.tme.lent.sh
  ```
- generate template data  + training model on Target side 
  ```bash
  bash ./scripts/run2.tme.lent.sh
  # bash ./scripts/run3.tme.probt.sh
  # bash ./scripts/run4.tme.dept.sh
  ```


    > **Note**: each scripts contain end2end all steps from: generate template => train source to target template (in run 2-4) => prepare data (in run 2-4) => train (template + source) to target. The step using stanford for generate syntactic tree is time consuming, if you want to skip this step, please use the data preprocessed data and **comment** this line in each run: 

    
    ```bash
    bash ./scripts/run_gen_template_dept.sh $DATA_DIR
    ```  

    > **Note**: There are alot of hyper-parameters in file `scripts/run_baseline.sh` or `scripts/run_templ.sh`, change it if you want to fine-tuning for larger model. The current script just suitable for small dataset - IWSLT. When you want to train on dataset WMT, you need to change some hyper parameters, for example: the number of head = 8, batch size = 8192, learning rate = 2.0 etc. 

    > **Note**: This code based on [Fairseq project](https://github.com/pytorch/fairseq) version 0.10.2, the main difference class for TME model is saved in `src/fairseqSyntaxNMT` 

##  License
MIT-licensed. 

## Citation

Please cite as:

``` bibtex
@proceedings{multisources-trans-template-nmt,
  editor    = {Phuong Nguyen and
                Tung Le and
                Thanh-Le Ha and
                Thai Dang and
                Khanh Tran and
                Kim Anh Nguyen and
                Nguyen Le Minh},
  title     = {Advances and Trends in Artificial Intelligence. Artificial Intelligence
               Practices - 35th International Conference on Industrial, Engineering
               and Other Applications of Applied Intelligent Systems, {IEA/AIE} 2022,
               Kitakyushu, Japan, July 19 – July 22, 2022},
  year      = {2022}
}
``` 