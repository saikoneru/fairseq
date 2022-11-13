#!/bin/bash

src=en
tgt=de
version=test
fairseq=/project/OML/skoneru/fairseq
bpe_data=/project/OML/skoneru/export_63/wmt14_de-en/bpe_test/
data_bin=$fairseq/data-bin/$version/git/wmt/$src-$tgt/
dict_path=$fairseq/data-bin/v1/git/wmt/$src-$tgt/
new_words=/project/OML/skoneru/export_63/scripts/adapt_fairseq/new_words.txt

mkdir -p $data_bin

rm ${data_bin}/dict*

for dir in "${src}-${tgt}" "${tgt}-${src}"; do
  IFS='-'
  read -ra src_tgt <<< "$dir"
  src_lang=${src_tgt[0]}
  tgt_lang=${src_tgt[1]}
  IFS=''
  fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
    --trainpref $bpe_data/train --validpref $bpe_data/valid --testpref $bpe_data/test \
    --destdir $data_bin --joined-dictionary --srcdict $dict_path/dict.$src.txt  \
    --extend-dict-file $new_words
done
