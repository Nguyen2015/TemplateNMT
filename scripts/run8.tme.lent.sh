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
# link script files
cd $DATA_DIR 
ln -s ../../scripts/run_templ.sh
ln -s ../../scripts/eval_templ.sh
cp ../../scripts/iwslt14deen-src-lent-setting.sh setting.sh
cd $ROOT_DIR

source activate ./nmt_py38_env/

# # --------------
# generate template
bash ./scripts/run_gen_template_lent.sh $DATA_DIR

# # --------------
# train + eval baseline model
cd $ROOT_DIR
echo $(pwd)
bash $DATA_DIR/run_templ.sh

 
