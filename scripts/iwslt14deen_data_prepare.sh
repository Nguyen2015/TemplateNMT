# prepare iwslt 2014 de-en original data 

CUR_DIR=$(pwd) 
cd ./scripts/ && wget https://raw.githubusercontent.com/pytorch/fairseq/master/examples/translation/prepare-iwslt14.sh && cd $CUR_DIR
cd ./data && bash ../scripts/prepare-iwslt14.sh && cd CUR_DIR

mv data/iwslt14.tokenized.de-en data/iwslt14deen
rm -r data/iwslt14deen/tmp

ls -lt data/iwslt14deen