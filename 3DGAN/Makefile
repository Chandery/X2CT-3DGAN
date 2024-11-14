train:
	python3 train.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml --gpu=2,3  --dataset=train --tag=2024-11-14-pengu  --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --valid_dataset=true
test:
	python3 test.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml --gpu=0  --dataset=test --tag=lr_3e-5_multi --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --check_point=1000 --how_many=3