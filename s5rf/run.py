from argparse import ArgumentParser, ArgumentError
from train import run_train

def str_to_bool(string: str) -> bool:
    if string == "True":
        return True
    elif string == "False":
        return False
    else:
        ArgumentError("Expected either \"True\" or \"False\".")


if __name__ == "__main__":
    parser = ArgumentParser()

    # dataset args
    parser.add_argument("--dataset", type=str, choices=["sMNIST", "psMNIST", "SHD", "SSC"], default="sMNIST", help="name of dataset")
    parser.add_argument("--batchsize", type=int, help="batchsize")
    parser.add_argument("--load_into_ram", type=str_to_bool, default=False, help="Only implemented for SSC. Speeds up dataloading. Requires more than 32 gb of RAM.")

    # model args
    parser.add_argument("--num_neurons", type=int, help="number of neurons for each layer")
    parser.add_argument("--num_blocks", type=int, help="number of Lambda blocks to initialize the RF neurons with")
    parser.add_argument("--num_layers", type=int, help="number of layers")
    parser.add_argument("--discretization", type=str, choices=["dirac", "zoh", "bilinear"], help="discretization method of the first layer")
    parser.add_argument("--eta_min", type=float, help="minimum eta value for initialization")
    parser.add_argument("--eta_max", type=float, help="maximum eta value for initialization")
    parser.add_argument("--keep_imag", type=str_to_bool, default=True, help="whether to propagate complex numbers between layers")

    # training args
    parser.add_argument("--seed", type=int, default=42, help="seed used")
    parser.add_argument("--epochs", type=int, help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate used in general")
    parser.add_argument("--lr_ssm", type=float, default=3e-4, help="learning rate of Lambda and eta")
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
    parser.add_argument("--weight_decay", type=float, default=0., help="weight decay of optimizer, not applied to Lambda and eta")
    parser.add_argument("--apply_cutmix", type=str_to_bool, default=False, help="whether cutmixing should be applied to batch before training")
    parser.add_argument("--apply_random_shift", type=str_to_bool, default=False, help="whether random shift should be applied to batch before training")

    # misc args
    parser.add_argument("--use_wandb", type=str_to_bool, default=False, help="whether to log with wandb")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")

    args = parser.parse_args()

    run_train(args)


