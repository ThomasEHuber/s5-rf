import jax
from .model.classifier import Classifier
from .dataloading import dataloaders
from .util.train_helpers import init_optimizer, train_epoch

def train(args) -> Classifier:
    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, model_key, train_key = jax.random.split(rng_key, num=3)
    if args.dataset == "sMNIST":
        train_dataloader, val_dataloader, test_dataloader = dataloaders.create_mnist_dataloaders(
            path=args.data_dir,
            batch_size=args.batchsize,
            permute=False
        )
        input_dim = 1
        output_dim = 10
    elif args.dataset == "psMNIST":
        train_dataloader, val_dataloader, test_dataloader = dataloaders.create_mnist_dataloaders(
            path=args.data_dir,
            batch_size=args.batchsize,
            permute=True
        )
        input_dim = 1
        output_dim = 10
    elif args.dataset == "SHD":
        train_dataloader, val_dataloader, test_dataloader = dataloaders.create_shd_dataloaders(
            path=args.data_dir,
            batch_size=args.batchsize,
        )
        input_dim = 140
        output_dim = 20
    elif args.dataset == "SSC":
        train_dataloader, val_dataloader, test_dataloader = dataloaders.create_ssc_dataloaders(
            path=args.data_dir,
            batch_size=args.batchsize,
        )
        input_dim = 140
        output_dim = 35
    

    model = Classifier(
        key=model_key, 
        input_dim=input_dim,
        output_dim=output_dim,
        num_neurons=[args.num_neurons] * args.num_layers,
        num_blocks=[args.num_blocks] * args.num_layers,
        dt_min=args.eta_min,
        dt_max=args.eta_max,
        activation="cartesian_spike",
        discretization=args.discretization,
        keep_imag=args.keep_imag,
        v_pos="str",
        apply_skip=True,
        dropout=args.dropout,
    )

    optim, opt_state = init_optimizer(
        model=model,
        standard_lr=args.lr,
        ssm_lr=args.lr_ssm,
        weight_decay=args.weight_decay,
        decay_steps=args.epochs * len(train_dataloader)
    )

    model = train_epoch(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        key=train_key,
        optim=optim,
        opt_state=opt_state,
        apply_cutmix=args.dataset == "SSC",
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )

    






