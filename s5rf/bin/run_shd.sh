python run.py --dataset=SHD --batchsize=32 --num_neurons=128 --num_blocks=32 \
              --num_layers=2 --discretization=dirac --eta_min=0.004 --eta_max=0.1 --keep_imag=True \
              --seed=42 --epochs=20 --lr=0.004 --lr_ssm=0.002 --dropout=0.1 --weight_decay=0.0001 \
              --apply_cutmix=False --apply_random_shift=True --use_wandb=False --wandb_project=test