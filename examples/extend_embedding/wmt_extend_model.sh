#!/bin/bash


ROOT=$(pwd "$0")
src=en
tgt=de
version=test
fairseq=/project/OML/skoneru/fairseq
data_bin=$fairseq/data-bin/test/git/wmt/$src-$tgt/
output_dir=$fairseq/checkpoints/test/
new_words=/project/OML/skoneru/export_63/scripts/adapt_fairseq/new_words.txt
seeds=( 1 )
    
    
for seed in "${seeds[@]}"; do 
  for dir in "${src}-${tgt}" "${tgt}-${src}"; do
    IFS='-'
    read -ra src_tgt <<< "$dir"
    src_lang=${src_tgt[0]}
    tgt_lang=${src_tgt[1]}
    save_dir=${output_dir}_${seed}/$dir/
    IFS=''
    mkdir -p $save_dir
    fairseq-train $data_bin/ --arch transformer \
     --dropout 0.2 --criterion label_smoothed_cross_entropy \
     --label-smoothing 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' \
      --lr 0.0001 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
     --warmup-init-lr '1e-07'  --max-tokens 12000  --save-dir $save_dir \
     --finetune-from-model $fairseq/checkpoints/v1/git/wmt/$dir/checkpoint_best.pt \
     --validate-interval 2  --seed $seed --update-freq 2 \
     --log-file $save_dir/train.log  \
     --no-epoch-checkpoints --fp16 \
     --patience 5 --extend-emb 
    wait
    exit
  done
done
