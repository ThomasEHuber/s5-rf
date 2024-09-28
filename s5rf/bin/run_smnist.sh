python run.py --dataset=sMNIST --batchsize=64 --num_neurons=128 --num_blocks=8 \
              --num_layers=2 --discretization=zoh --eta_min=0.001 --eta_max=0.1 --keep_imag=True \
              --seed=42 --epochs=100 --lr=0.008 --lr_ssm=0.002 --dropout=0.15 --weight_decay=0.0001 \
              --apply_cutmix=False --apply_random_shift=False --use_wandb=True --wandb_project=test