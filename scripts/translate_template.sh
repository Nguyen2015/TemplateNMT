PATH_DATA_=`dirname "$0"`

PATH_DATA=${1:-$PATH_DATA_}
TYPE_TEMPL=${2:-"lent"}
RUN_PREDICT=true

echo $PATH_DATA
MOSES_LIB="$PATH_DATA/../mosesdecoder"

MAX_LEN_B=200

source $PATH_DATA/setting.sh 

for TYPE_F in "valid"  "train" "test"
do 
    TEST_FILE="${PATH_DATA}/${TYPE_F}.${TGT_LANG}"

    if $RUN_PREDICT ; then
        PATH_DATA_BIN_INFER=$PATH_DATA/infer-${TYPE_F}
        mkdir $PATH_DATA_BIN_INFER 

        cp ${PATH_DATA_BIN}/dict.$SRC_LANG.txt ${PATH_DATA_BIN}/dict.$TGT_LANG.txt $PATH_DATA_BIN_INFER

        # Binarize the dataset
        fairseq-preprocess \
            --source-lang $SRC_LANG --target-lang $TGT_LANG \
            --testpref $PATH_DATA/${TYPE_F} \
            --destdir $PATH_DATA_BIN_INFER --thresholdtgt 0 --thresholdsrc 0 \
            --workers 32 --only-source --srcdict $PATH_DATA_BIN_INFER/dict.$SRC_LANG.txt 


        fairseq-generate ${PATH_DATA_BIN_INFER} --path ${PATH_DATA}/averaged.pt  --beam 5 --max-tokens 16384 --max-len-b $MAX_LEN_B \
        | grep '^H' | sed 's/^H\-//g' | sort -t ' ' -k1,1 -n | cut -f 3- > ${PATH_DATA_BIN_INFER}/generated_best.result

        cp ${PATH_DATA_BIN_INFER}/generated_best.result ${PATH_DATA_BIN_INFER}/../generated_best.${TYPE_F}.result
        rm -r ${PATH_DATA_BIN_INFER}
    fi 
    
    cat ${PATH_DATA}//generated_best.${TYPE_F}.result | ${MOSES_LIB}/scripts/generic/multi-bleu.perl ${TEST_FILE} \
        > ${PATH_DATA}/log_avg_multi-bleu.${TYPE_F}.log

 
    mv ${PATH_DATA}//generated_best.${TYPE_F}.result ${PATH_DATA}/$TYPE_F.${TGT_LANG/parsed/pred}
done
