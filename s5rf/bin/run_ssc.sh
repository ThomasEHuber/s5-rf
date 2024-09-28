python run.py --dataset=SSC --batchsize=128 --num_neurons=512 --num_blocks=16 \
              --num_layers=4 --discretization=dirac --eta_min=0.004 --eta_max=0.1 --keep_imag=True \
              --seed=42 --epochs=80 --lr=0.004 --lr_ssm=0.004 --dropout=0.3 --weight_decay=0.0001 \
              --apply_cutmix=True --apply_random_shift=True --use_wandb=True --wandb_project=test \
              --load_into_ram=False