RUN_PREDICT=${1:-true}

PATH_DATA=`dirname "$0"` # `dirname "$(readlink -f "$0")"`
echo $PATH_DATA

source $PATH_DATA/setting.sh 
echo $TEMPLATE_TYPE


MOSES_LIB="$PATH_DATA/../mosesdecoder"
TEST_FILE="$PATH_DATA/test.$TGT_LANG.debpe"
USER_DIR="src/fairseqSyntaxNMT"
 

 

# # run avg last 5 checkpoint 
if $RUN_PREDICT ; then
    python src/avg_last_checkpoint.py --inputs ${PATH_DATA} --num-epoch-checkpoints 5 --output ${PATH_DATA}/averaged.pt
    fairseq-generate ${PATH_DATA_BIN} \
            --source-lang $SRC_LANG --target-lang $TGT_LANG \
            --task template_translation --template-type $TEMPLATE_TYPE --user-dir $USER_DIR \
            --path ${PATH_DATA}/averaged.pt  --beam 5 --remove-bpe   > ${PATH_DATA}/generated.result.raw.log 
    cat ${PATH_DATA}/generated.result.raw.log | grep '^H' | sed 's/^H\-//g' | sort -t ' ' -k1,1 -n | cut -f 3- > ${PATH_DATA}/generated.result
    cat ${PATH_DATA}/generated.result.raw.log | grep '^T' | sed 's/^T\-//g' | sort -t ' ' -k1,1 -n | cut -f 2- > ${TEST_FILE} 
fi
cat ${PATH_DATA}/generated.result | ${MOSES_LIB}/scripts/generic/multi-bleu.perl ${TEST_FILE} > ${PATH_DATA}/log_avg_multi-bleu.log
cat  ${PATH_DATA}/log_avg_multi-bleu.log
 