python blend.py \
       --source_file data/source.png \
       --mask_file data/mask.png \
       --target_file data/target.png \
       --output_dir results/replacement \
       --ss 512 --ts 512 --x 256 --y 256 \
       --gpu 0  --num_steps 1000 --save_video True

python blend.py \
       --source_file data/3_source.png \
       --mask_file data/3_mask.png \
       --target_file data/3_target.png \
       --output_dir results/3 \
       --ss 384 --ts 512 --x 320 --y 256 \
       --gpu 0  --num_steps 1000 --save_video True

python blend.py \
       --source_file data/5_source.png \
       --mask_file data/5_mask.png \
       --target_file data/5_target.png \
       --output_dir results/5 \
       --ss 320 --ts 512 --x 160 --y 256 \
       --gpu 0  --num_steps 1000 --save_video True





