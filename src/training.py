import logging
from modulefinder import Module
import numpy as np
import torch
from torch import Tensor, nn 
from pathlib import Path
import shutil
from helps import load_config,make_model_dir,make_logger, log_cfg
from helps import set_seed, parse_train_arguments
from model import build_model
from torch.utils.tensorboard import SummaryWriter
from builders import build_gradient_clipper, build_optimizer
import heapq

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


class TrainManager(object):
    """
    Manage the whole training loop, valiation, learning rate scheduling and early stopping.
    """
    def __init__(self, model:Module, cfg: dict) -> None:

        (model_dir, loss_type, label_smoothing,
        normalization, learning_rate_min, keep_best_ckpts,
        logging_freq, validation_freq, log_valid_sentences,
        early_stopping_metric, shuffle, epochs, max_updates,
        batch_size, batch_type, random_seed,
        device, n_gpu, num_workers,
        reset_best_ckpt, reset_scheduler,
        reset_optimizer, reset_iter_state) = parse_train_arguments(train_cfg=cfg["training"])

        self.model_dir = model_dir
        self.logging_freq = logging_freq
        self.validation_freq = validation_freq
        self.log_valid_sentences = log_valid_sentences
        # FIXME tensorboard how to use with pytorch
        self.tb_writer = SummaryWriter(log_dir=(model_dir/"tensorboard").as_posix())

        # model
        self.model = model
        self.model.log_parameters_list()
        self.model.loss_function = (loss_type, label_smoothing)
        logger.info(self.model)

        # CPU/GPU
        self.device = device
        self.n_gpu = n_gpu
        self.num_workers = num_workers
        if self.device.type == "cuda":
            self.model.to(self.device)
        
        # optimization
        self.clip_grad_fun = build_gradient_clipper(train_cfg=cfg["training"])
        self.optimizer = build_optimizer(train_cfg=cfg["training"], parameters=self.model.parameters())

        # save/delete checkpoints
        self.num_ckpts = keep_best_ckpts
        self.ckpt_queue = [] # heap queue      List[Tuple[float, Path]]

        # early_stopping
        self.early_stopping_metric = early_stopping_metric
        if self.early_stopping_metric in ["ppl", "loss"]:
            self.minimize_metric = True  # lower is better
        elif self.early_stopping_metric in ["acc", "bleu"]:
            self.minimize_metric = False # higher is better

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler()

        # data & batch handling
        self.seed = random_seed
        self.shuffle = shuffle
        self.epochs = epochs
        self.max_updates = max_updates
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.learning_rate_min = learning_rate_min
        self.normalization = normalization

        # initialize training statistics
        self.stats = self.TrainStatistics(
            steps=0, is_min_lr=False, is_max_updates=False,
            total_tokens=0, best_ckpt_iter=0, minimize_metric = self.minimize_metric,
            best_ckpt_score=np.inf if self.minimize_metric else -np.inf,
        )

    class TrainStatistics:

        def __init__(self, steps:int=0, is_min_lr:bool=False,
                     is_max_update:bool=False, total_tokens:int=0,
                     best_ckpt_iter:int=0, best_ckpt_score: float=np.inf,
                     minimize_metric: bool=True) -> None:
            self.steps = steps 
            self.is_min_lr = is_min_lr
            self.is_max_update = is_max_update
            self.total_tokens = total_tokens
            self.best_ckpt_iter = best_ckpt_iter
            self.best_ckpt_score = best_ckpt_score
            self.minimize_metric = minimize_metric
        
        def is_best(self, score) -> bool:
            if self.minimize_metric:
                is_best = score < self.best_ckpt_score
            else: 
                is_best = score > self.best_ckpt_score
            return is_best
        
        def is_better(self, score: float, heap_queue: list):
            assert len(heap_queue) > 0
            if self.minimize_metric:
                is_better = score < heapq.nlargest(1, heap_queue)[0][0]
            else:
                is_better = score > heapq.nsmallest(1, heap_queue)[0][0]
            return is_better











if __name__ == "__main__":
    cfg_file = "configs/transformer.yaml"
    train(cfg_file=cfg_file)