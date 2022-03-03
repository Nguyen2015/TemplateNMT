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


source activate ./nmt_py38_env/

# # --------------
# link script files
cd $DATA_DIR 
ln -s ../../scripts/run_baseline.sh
ln -s ../../scripts/eval_translation_baseline.sh
cp ../../scripts/iwslt14deen-setting.sh setting.sh


# # --------------
# train + eval baseline model
cd $ROOT_DIR
echo $(pwd)
bash $DATA_DIR/run_baseline.sh

 
