
DATA_DIR=${1:-"./"}
TEMPLATE_SRC_SIDE=${2:-true}
CUR_DIR=$(pwd)


source $DATA_DIR/setting.sh

if $TEMPLATE_SRC_SIDE; then
    echo "generate template in source side ..."
    TEMPL_LANG=$SRC_LANG
else
    echo "generate template in target side ..."
    TEMPL_LANG=$TGT_LANG
fi

echo ${DATA_DIR}/*.$TEMPL_LANG

# link data
cd $DATA_DIR
ln -s train.$TEMPL_LANG train.$TEMPL_LANG.bpe
ln -s valid.$TEMPL_LANG valid.$TEMPL_LANG.bpe
ln -s test.$TEMPL_LANG test.$TEMPL_LANG.bpe
cd $CUR_DIR

# parse tree
python src/syntactic_tree_parser.py \
--path_folder_or_pattern "${DATA_DIR}/*.$TEMPL_LANG" \
--path_stanford_lib ./nmt_py38_env/stanford-corenlp-4.0.0/ \
--lang $TEMPL_LANG 

# generate dynamic template from constituent tree 
python src/template_generator.py \
--path_folder_or_pattern "${DATA_DIR}/" \
--type_template d3 --depth_level 3 \
 

# rename file name for correct pattern training data 
python src/prepare_data_supporter.py \
--path_folder ${DATA_DIR}

for TYPE_F in "train" "valid" "test"; do
    mv $DATA_DIR/$TYPE_F.$TEMPL_LANG.softtemplate $DATA_DIR/$TYPE_F.${TEMPL_LANG}_parsed_dept
done