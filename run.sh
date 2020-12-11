python -W ignore eval_align.py --data_path=data/dbp_wd_15k_V1/mapping/0_3/ --struct="[0,4,10,1,1,0,0,1,1,0,1]" --learning_rate=0.000068 --L2=0.001598 --drop=0.26 --batch_size=512 --decay_rate=0.99715;

python -W ignore eval_align.py --data_path=data/dbp_yg_15k_V1/mapping/0_3/ --struct="[1,11,2,2,1,0,1,1,1,0,1]" --learning_rate=0.00003 --L2=0.001378 --drop=0.45 --batch_size=512 --decay_rate=0.989504;

python -W ignore eval_align.py --data_path=data/en_fr_15k_V1/mapping/0_3/  --struct="[0,2,7,0,0,0,0,0,0,0,0]"  --learning_rate=0.000071 --L2=0.001332 --drop=0.31 --batch_size=512 --decay_rate=0.98616;

python -W ignore eval_align.py --data_path=data/en_de_15k_V1/mapping/0_3/  --struct="[2,2,7,1,0,0,0,1,0,0,0]"  --learning_rate=0.000103 --L2=0.004069 --drop=0.25 --batch_size=2048 --decay_rat=0.98126;

python -W ignore eval_align.py --data_path=data/dbp_wd_15k_V2/mapping/0_3/ --struct="[0,10,7,0,1,0,0,0,0,0,0]" --learning_rate=0.000085 --L2=0.001761 --drop=0.11 --batch_size=1024 --decay_rate=0.9864;

python -W ignore eval_align.py --data_path=data/dbp_yg_15k_V2/mapping/0_3/ --struct="[2,10,2,2,0,0,0,0,0,0,0]" --learning_rate=0.000051 --L2=0.000521 --drop=0.39 --batch_size=512 --decay_rate=0.98797;

python -W ignore eval_align.py --data_path=data/en_fr_15k_V2/mapping/0_3/  --struct="[0,2,7,0,0,0,0,0,0,0,0]"  --learning_rate=0.000062 --L2=0.000983 --drop=0.29 --batch_size=1024 --decay_rate=0.9894;

python -W ignore eval_align.py --data_path=data/en_de_15k_V2/mapping/0_3/  --struct="[0,2,7,0,0,0,0,0,0,0,0]"  --learning_rate=0.000113 --L2=0.001397 --drop=0.21 --batch_size=1024 --decay_rate=0.9848;




python -W ignore eval_link.py --data_path=data/WN18RR/     --struct="[0,0,2,0,0,0,0,0,0,0,0]"  --learning_rate=0.003093 --L2=0.000004 --drop=0.35 --batch_size=2048 --decay_rate=0.9876;

python -W ignore eval_link.py --data_path=data/FB15K237/  --struct="[2,1,11,1,1,0,0,0,0,0,0]"  --learning_rate=0.000634 --L2=0.0000001 --drop=0.42 --batch_size=2048 --decay_rate=0.9824;
