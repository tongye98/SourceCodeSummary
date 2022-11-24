import argparse
from src.training import train
from src.testing import test
from src.build_database import build_database
from src.build_code_semantic import build_code_semantic_database
from src.retrieval_training import retrieval_train
from src.retrieval_testing import retrieval_test
from src.rencos_test import rencos_test
from src.metrics import test_metrics

def main():
    parser = argparse.ArgumentParser("Transformer")

    parser.add_argument("mode", choices=["train","test","build_database","build_code_semantic","retrieval_train","retrieval_test","rencos_test", "test_metrics"])
    parser.add_argument("config_path", type=str, help="path to a config yaml file")
    parser.add_argument("--ckpt", type=str, help="model checkpoint for prediction")
    parser.add_argument("--hidden_representation_path", type=str, help="where to store the hidden state")
    parser.add_argument("--token_map_path", type=str, help="where to store the corresponding token id")
    parser.add_argument("--index_path", type=str, help="where to store faiss index")
    parser.add_argument("--data_dtype", type=str, choices=['float16', 'float32'],help="how to store hidden representaion")

    args = parser.parse_args()

    if args.mode == "train":
        train(cfg_file=args.config_path)

    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt_path=args.ckpt)

    elif args.mode == "build_database":
        build_database(cfg_file=args.config_path)
    
    elif args.mode == "build_code_semantic":
        build_code_semantic_database(cfg_file=args.config_path)

    elif args.mode == "retrieval_train":
        retrieval_train(cfg_file=args.config_path)

    elif args.mode == "retrieval_test":
        retrieval_test(cfg_file=args.config_path, ckpt_path=args.ckpt)

    elif args.mode == "rencos_test":
        rencos_test(cfg_file=args.config_path, ckpt_path=args.ckpt)
    
    elif args.mode == "test_metrics":
        test_metrics()
        
    else:
        raise ValueError("Unkonwn mode!")

if __name__ == "__main__":
    main()