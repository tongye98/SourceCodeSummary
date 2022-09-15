import argparse
from src.training import train
from src.prediction import test 
from src.build_database import build_database

def main():
    parser = argparse.ArgumentParser("Transformer")

    parser.add_argument("mode", choices=["train","test","build_database"], help="Train a model or Test.")
    parser.add_argument("config_path", type=str, help="path to a config yaml file.")
    parser.add_argument("-c","--ckpt", type=str, help="model checkpoint for prediction.")
    parser.add_argument("-o","--output_path", type=str, help="path for saving test result.")
    parser.add_argument("-attention","--save_attention", action="store_true", help="save attenton visualization")
    parser.add_argument("-s","--save_scores", action="store_true", help="save log_probability scores.")
    parser.add_argument("--skip_test", action="store_true", help="Skip test after training.")
    parser.add_argument("--embedding_path", type=str, help="where to store the hidden state")
    parser.add_argument("--token_map_path", type=str, help="where to store the corresponding token id")
    parser.add_argument("--index_path", type=str, help="where to store faiss index")

    args = parser.parse_args()

    if args.mode == "train":
        train(cfg_file=args.config_path, skip_test=args.skip_test)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt_path=args.ckpt, output_path=args.output_path)
    elif args.mode == "build_database":
        build_database(cfg_file=args.config_path, division="train", ckpt=args.ckpt,
         embedding_path=args.embedding_path, token_map_path=args.token_map_path, index_path=args.index_path)
    else:
        raise ValueError("Unkonwn mode!")

if __name__ == "__main__":
    main()