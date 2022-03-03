DATA_DIR="data/iwslt14deen"
ROOT_DIR=$(pwd)

if ! [ -d $DATA_DIR ] ; then 
    echo "data $DATA_DIR doesn't exist..."
    exit 1
fi

if ! [ -d "data/mosesdecoder" ] ; then 
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git data/mosesdecoder
fi


if ! [ -f "./nmt_py38_env/stanford-corenlp-4.0.0/stanford-german-corenlp-0000-00-00-models.jar" ] ; then 
    echo 'Download models stanford-corenlp germany model ... '
    wget http://nlp.stanford.edu/software/stanford-corenlp-4.0.0-models-german.jar
    mv stanford-corenlp-4.0.0-models-german.jar "./nmt_py38_env/stanford-corenlp-4.0.0/stanford-german-corenlp-0000-00-00-models.jar"
fi


# # --------------
# train step 1 - train a template generator NMT model 
# # --------------
source activate ./nmt_py38_env/

# # --------------
# generate template
cp scripts/iwslt14deen-setting.sh $DATA_DIR/setting.sh
bash ./scripts/run_gen_template_lent.sh $DATA_DIR false # second parameter [false] indicate the template is parsed in target side


# # --------------
# link script files
cd $DATA_DIR 
ln -s ../../scripts/run_baseline.sh
ln -s ../../scripts/eval_translation_baseline.sh
cp ../../scripts/iwslt14deen-tgt-lent-step1-setting.sh setting.sh
cd $ROOT_DIR
 
# #  train a template genearator NMT model 
rm $DATA_DIR/checkpoint_last.pt
bash $DATA_DIR/run_baseline.sh


# # --------------
# generate template 
# # --------------
bash ./scripts/translate_template.sh $DATA_DIR "lent"


# # --------------
# train step 2 - train NMT model using template  
# # --------------
# link script files
cd $DATA_DIR 
ln -s ../../scripts/run_templ.sh
ln -s ../../scripts/eval_templ.sh
cp ../../scripts/iwslt14deen-tgt-lent-step2-setting.sh setting.sh
cd $ROOT_DIR


# train + eval TME model
cd $ROOT_DIR
echo $(pwd)
rm $DATA_DIR/checkpoint_last.pt
bash $DATA_DIR/run_templ.sh

 
