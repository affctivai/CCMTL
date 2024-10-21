seed_path='/mnt/data/members/fusion/SEED/ExtractedFeatures'
save_file_name='results.csv'

python main.py --w_mode w --model_type CCMTL --dataset_dir $seed_path --n_classes 3 --save_file_name $save_file_name --reduction_ratio 2 --lstm --modulator
