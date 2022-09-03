import argparse
from src.training import train
from src.prediction import test 

def main():
    parser = argparse.ArgumentParser("Transformer")

    parser.add_argument("mode", choices=["train","test"], help="Train a model or Test.")
    parser.add_argument("config_path", type=str, help="path to a config yaml file.")
    parser.add_argument("-c","--ckpt", type=str, help="model checkpoint for prediction.")
    parser.add_argument("-o","--ouput_path", type=str, help="path for saving test result.")
    parser.add_argument("-attention","--save_attention", action="store_true", help="save attenton visualization")
    parser.add_argument("-s","--save_scores", action="store_true", help="save log_probability scores.")
    parser.add_argument("--skip_test", action="store_true", help="Skip test after training.")

    args = parser.parse_args()

    if args.mode == "train":
        train(cfg_file=args.config_path, skip_test=args.skip_test)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt_path=args.ckpt, output_path=args.output_path)
    else:
        raise ValueError("Unkonwn mode!")

if __name__ == "__main__":
    main()