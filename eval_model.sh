#!/bin/bash

src=en
tgt=de
split=random
config=gnome/$src-$tgt/$split/wmt_finetune/
bpe_dir=/export/data1/skoneru/data/matched/$src-$tgt/$split/bpe/
raw_dir=/export/data1/skoneru/data/filtered/$split/raw/


data_dir=data-bin/$config/
src_file=$bpe_dir/test.$src

model_dir=checkpoints/$config/tmp/update8_0
model_path="$model_dir/checkpoint_avg.pt"

out_dir=hypothesis/$config/tmp/update8_0
hyp_file=$out_dir/hyp.$tgt

ref_file_orig=$raw_dir/test.$tgt
MOSES=./mosesdecoder/scripts
charpy=CharacTER/CharacTER.py
ref_file=$out_dir/ref.$tgt

mkdir -p $out_dir
cp $ref_file_orig $ref_file

fairseq-interactive $data_dir --path $model_path --beam 5 --source-lang $src --target-lang $tgt --input $src_file --buffer-size 128 --batch-size 32 --remove-bpe=subword_nmt > $out_dir/pred.log
wait

if [ ! -d "$MOSES" ]; then
	git clone https://github.com/moses-smt/mosesdecoder.git
fi

grep ^H $out_dir/pred.log | cut -f3 > $hyp_file.tok
$MOSES/tokenizer/detokenizer.perl -l $tgt < $hyp_file.tok > $hyp_file

cat $hyp_file | sacrebleu $ref_file -m bleu -b -w 4
python CharacTER/CharacTER.py -r $ref_file -o $hyp_file

conda deactivate
conda activate base
comet-score -s $raw_dir/test.$src -t $hyp_file -r $ref_file --quiet
conda deactivate
conda activate fairseq
