from argparse import ArgumentParser
from .train import train

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="seed used")
    parser.add_argument("--epochs", type=int, help="number of epochs to train for")

    parser.add_argument("--dataset", type=str, choices=["sMNIST", "psMNIST", "SHD", "SSC"], default="sMNIST", help="name of dataset")
    parser.add_argument("--data_dir", type=str, default="../data", help="path where data is stored")
    parser.add_argument("--batchsize", type=int, help="batchsize")

    parser.add_argument("--num_neurons", type=int, help="number of neurons for each layer")
    parser.add_argument("--num_blocks", type=int, help="number of Lambda blocks to initialize the RF neurons with")
    parser.add_argument("--num_layers", type=int, help="number of layers")
    parser.add_argument("--discretization", type=str, choices=["dirac", "zoh", "bilinear"], help="discretization method of the first layer")
    parser.add_argument("--eta_min", type=int, help="minimum eta value for initialization")
    parser.add_argument("--eta_max", type=int, help="maximum eta value for initialization")
    parser.add_argument("--keep_imag", type=bool, default=True, help="whether to propagate complex numbers between layers")

    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate used in general")
    parser.add_argument("--lr_ssm", type=float, default=3e-4, help="learning rate of Lambda and eta")
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
    parser.add_argument("--weight_decay", type=float, default=0., help="weight decay of optimizer, not applied to Lambda and eta")

    parser.add_argument("use_wandb", type=bool, default=False, help="whether to log with wandb")
    parser.add_argument("wandb_project", type=str, default=None, help="wandb project name")

    args = parser.parse_args()

    train(args)


