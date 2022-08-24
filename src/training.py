import logging
from unittest import SkipTest, skip 
import torch
from torch import Tensor, nn 
from pathlib import Path
import shutil
from helps import load_config,make_model_dir,make_logger, log_cfg
from helps import set_seed
from model import build_model

logger = logging.getLogger(__name__) 


def train(cfg_file: str, skip_test:bool=False) -> None:
    """
    Training function. After training, also test on test data if given.
    """
    cfg = load_config(Path(cfg_file))

    # make model dir
    model_dir = make_model_dir(Path(cfg["training"]["model_dir"]),overwrite=cfg["training"].get("overwrite",False))

    # make logger
    make_logger(model_dir, mode="train")

    # write all entries of config to the log file.
    log_cfg(cfg)

    # store copy of origianl training config in model dir. 
    # as_posix change path separator to unix "/"
    shutil.copy2(cfg_file, (model_dir/"config.yaml").as_posix())

    # set the whole random seed
    set_seed(seed=int(cfg["training"].get("random_seed", 980820)))

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(data_cfg=cfg["data"])

    # store the vocabs and tokenizers
    # TODO

    # build an transformer(encoder-decoder) model
    model = build_model(model_cfg=cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # for training management.
    trainer = TrainManager(model=model, cfg=cfg)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    # after train, let's test on test data.
    if not skip_test:
        # TODO
        pass
        # predict with best model on validation and test data.
    else:
        logger.info("Skipping test after training the model!")












if __name__ == "__main__":
    cfg_file = "configs/transformer.yaml"
    train(cfg_file=cfg_file)