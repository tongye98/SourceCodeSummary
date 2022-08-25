import logging
from modulefinder import Module
from os import symlink
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from pathlib import Path
import shutil
from typing import List
from helps import load_config,make_model_dir,make_logger, log_cfg
from helps import set_seed, parse_train_arguments, load_model_checkpoint
from helps import symlink_update, delete_ckpt, write_validation_output_to_file
from prediction import predict
from src.data import load_data, make_data_iter
from model import build_model
from torch.utils.tensorboard import SummaryWriter
from builders import build_gradient_clipper, build_optimizer
import heapq
import math
import time

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
        device, n_gpu, num_workers,load_model,
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

        # for train data sampler.
        self.train_iter, self.train_iter_state = None, None

        # initialize training statistics
        self.stats = self.TrainStatistics(
            steps=0, is_min_lr=False, is_max_updates=False,
            total_tokens=0, best_ckpt_iter=0, minimize_metric = self.minimize_metric,
            best_ckpt_score=np.inf if self.minimize_metric else -np.inf,
        )
        
        if load_model is not None:
            self.init_from_checkpoint(load_model,
                reset_best_ckpt=reset_best_ckpt, reset_scheduler=reset_scheduler,
                reset_optimizer=reset_optimizer, reset_iter_state=reset_iter_state)
        
        # multi-gpu training
        # TODO 

        # config for generation
        self.valid_cfg = cfg["testing"].copy()
        self.valid_cfg["beam_size"] = 1 # 1 means greedy decoding during train loop
        self.valid_cfg["batch_size"] = self.batch_size
        self.valid_cfg["batch_type"] = self.batch_type
        self.valid_cfg["n_best"] == 1   # only the best one
        
        self.valid_cfg["generate_unk"] = True

    def save_model_checkpoint(self, new_best:bool, score:float) -> None:
        """
        Save model's current parameters and the training state to a checkpoint.
        The training state contains the total number of training steps, the total number of
        training tokens, the best checkpoint score and iteration so far, and optimizer and scheduler states.

        new_best: for update best.ckpt
        score: Validation score which is used as key of heap queue.
        """
        model_path = Path(self.model_dir) / f"{self.stats.steps}.ckpt"
        # FIXME for multi gpu
        model_state_dict = self.model.state_dict()
        global_state = {
            "steps": self.stats.steps,
            "total_tokens": self.stats.total_tokens,
            "best_ckpt_score": self.stats.best_ckpt_score,
            "best_ckpt_iteration": self.stats.best_ckpt_iter,
            "model_state": model_state_dict,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "train_iter_state":self.train_iter.batch_sampler.sampler.generator.get_state(),
        }
        torch.save(global_state, model_path.as_posix())
        
        # how to update queue and keep the best number of ckpt.
        symlink_target = Path(f"{self.stats.steps}.ckpt")
        last_path = Path(self.model_dir) / "latest.ckpt"
        prev_path = symlink_update(symlink_target, last_path)
        best_path = Path(self.model_dir) / "best.ckpt"
        if new_best:
            prev_path = symlink_update(symlink_target, best_path)
            assert best_path.resolve().stem == str(self.stats.best_ckpt_iter)
        
        # push and pop from the heap quene.
        to_delete = None
        if not math.isnan(score) and self.num_ckpts > 0:
            if len(self.ckpt_queue) < self.num_ckpts: # no need pop, only push
                heapq.heappush(self.ckpt_queue, (score, model_path))
            else: # first pop the worst, then push
                if self.minimize_metric: # smaller, better
                    heapq.heapify(self.ckpt_queue)
                    to_delete = heapq._heapify_max(self.ckpt_queue)
                    heapq.heappush(self.ckpt_queue,(score, model_path))
                else: # bigger, better
                    to_delete = heapq.heappushpop(self.ckpt_queue,(score, model_path))
            
            if to_delete is not None:
                assert to_delete[1] != model_path # don't delete the last ckpt
                if to_delete[1].stem != best_path.resolve().stem:
                    delete_ckpt(to_delete[1])  # don't delete the best ckpt
            
            assert len(self.ckpt_queue) <= self.num_ckpts

            if prev_path is not None and prev_path.stem not in [c[1].stem for c in self.ckpt_queue]:
                delete_ckpt(prev_path)

    def init_from_checkpoint(self, path=Path, 
                             reset_best_ckpt:bool=False, reset_scheduler:bool=False,
                             reset_optimizer:bool=False, reset_iter_state:bool=False):
        """
        Initialize the training from a given checkpoint file.
        The checkpoint file contain not only model parameters, but also
        scheduler and optimizer states.
        """
        logger.info("Loading model from %s", path)
        model_checkpoint = load_model_checkpoint(path=path, device=self.device)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["model_state"])
        else:
            logger.info("Reset Optimizer.")
        
        if not reset_scheduler:
            if model_checkpoint["scheduler_state"] is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
        else:
            logger.info("Reset Scheduler.")
        
        if not reset_best_ckpt:
            self.stats.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.stats.best_ckpt_iter = model_checkpoint["best_ckpt_iteration"]
        else:
            logger.info("Reset tracking of the best checkpoint.")
        
        if not reset_iter_state:
            assert "train_iter_state" in model_checkpoint
            self.stats.steps = model_checkpoint["steps"]
            self.stats.total_tokens = model_checkpoint["total_tokens"]
            self.train_iter_state = model_checkpoint["train_iter_state"]
        else:
            logger.info("Reset data iterator (random seed: {%d}).", self.seed)
        
        if self.device.type == "cuda":
            self.model.to(self.device)
        
    def train_and_validate(self, train_data:Dataset, valid_data:Dataset) -> None:
        """
        Train the model and validate it from time to time on the validation set.
        """
        self.train_iter = make_data_iter()
        if self.train_iter_state is not None:
            self.train_iter.batch_sampler.sampler.generator.set_state(
                self.train_iter_state.cpu())
        
        # train and validate main loop
        logger.info("Train stats:\n"
                    "\tdevice: %s\n"
                    "\tn_gpu: %d\n"
                    "\tbatch_size per device: %d\n",
                    self.device.type, self.n_gpu,
                    self.batch_size // self.n_gpu,)
        try:
            for epoch_no in range(self.epochs):
                logger.info("Epoch %d", epoch_no + 1)

                self.model.train()
                self.model.zero_grad()

                if self.scheduler_step_at == "epoch":
                    self.scheduler.step(epoch=epoch_no)
                
                # Statistic for each epoch.
                start_time = time.time()
                total_valid_duration_time = 0
                start_tokens = self.stats.total_tokens
                epoch_loss = 0
                total_batch_loss = 0

                for i, batch_data in enumerate(self.train_iter):
                    batch_loss = self.train_step(batch_data)
                    total_batch_loss += batch_loss

                    # clip gradients (in-place)
                    if self.clip_grad_fun is not None:
                        self.clip_grad_fun(parameters=self.model.parameters())
                    
                    # make gradient step
                    # Note: loss.backward() inside self.train_step()
                    self.optimizer.step()

                    # decay learning_rate(lr)
                    # FIXME why not under logging ?
                    if self.scheduler_step_at == "step":
                        self.scheduler.step(self.stats.steps)
                    
                    # reset gradients
                    self.model.zero_grad()

                    # increment step counter
                    self.stats.steps += 1
                    if self.stats.steps >= self.max_updates:
                        self.stats.is_max_update == True
                    
                    # log learning process and write tensorboard
                    if self.stats.steps % self.logging_freq == 0:
                        elapse_time = time.time() - start_time - total_valid_duration_time
                        elapse_token_num = self.stats.total_tokens - start_tokens
                        # FIXME why is total batch loss
                        self.tb_writer.add_scalar("Train/batch_loss", total_batch_loss, self.stats.steps)
                    
                        logger.info("Epoch %3d, Step: %8d, Batch Loss: %12.6f, Lr: %.6f, Tokens per sec: %8.0f",
                        epoch_no + 1, self.stats.steps, total_batch_loss, self.optimizer.param_groups[0]["lr"],
                        elapse_token_num / elapse_time)

                        start_time = time.time()
                        start_tokens = self.stats.total_tokens
                        # one log process may include more than one validation, so need total valid time.
                        total_valid_duration_time = 0
                    
                    # validate on the entire dev dataset
                    if self.stats.steps % self.validation_freq == 0:
                        valid_duration_time = self.validate(valid_data)
                        total_valid_duration_time += valid_duration_time
                    
                    # check current leraning rate(lr)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    if current_lr < self.learning_rate_min:
                        self.stats.is_min_lr = True 
                    
                    self.tb_writer.add_scalar("Train/learning_rate",current_lr, self.stats.steps)
                
                # check after a whole epoch.
                if self.stats.is_min_lr or self.stats.is_max_update:
                    log_string = (f"minimum learning rate {self.learning_rate_min}"
                                    if self.stats.is_min_lr else 
                                    f"maximun number of updates(steps) {self.max_updates}")
                    logger.info("Training enede since %s was reached!", log_string)
                    break 
                    
                logger.info("Epoch %3d: total training loss %.2f", epoch_no + 1, epoch_loss)
            else: # normal ended after training.
                logger.info("Training ended after %3d epoches!", epoch_no + 1)

            logger.info("Best Validation result (greedy) at step %8d: %6.2f %s.",
                        self.stats.best_ckpt_iter, self.stats.best_ckpt_score, self.early_stopping_metric)
                    
        except KeyboardInterrupt:
            self.save_model_checkpoint(False, float("nan"))

        return None 

    def train_step(self, batch_data):
        """
        Train the model on one batch: compute loss
        """
        # reactivate training.
        self.model.train()

        # get loss (run as during training with teacher forcing)
        batch_loss, log_probs = self.model(return_type="loss", src_input=None, trg_input=None,
                   src_mask=None, trg_mask=None, encoder_output = None)
        
        # FIXME should normalizer batch_loss ?

        batch_loss.backward()

        # increment token counter
        self.stats.total_tokens += batch_data.ntokens

        return batch_loss.item()

    def validate(self, valid_data: Dataset):
        """
        Validate on the valid dataset.
        return the validate time.
        """
        validate_start_time = time.time()
        (valid_scores, valid_references, valid_hypotheses, 
         valid_hypotheses_raw, valid_sequence_scores, 
         valid_attention_scores,) = predict(model=self.model, data=valid_data, compute_loss=True,
                                            device=self.device, n_gpu=self.n_gpu, 
                                            normalization=self.normalization, cfg=self.valid_cfg)
        valid_duration_time = time.time() - validate_start_time
        
        # write eval_metric and corresponding score to tensorboard
        for eval_metric, score in valid_scores.items():
            if not math.isnan(score):
                self.tb_writer.add_scalar(f"Valid/{eval_metric}", score, self.stats.steps)
        
        # the most important metric
        ckpt_score = valid_scores[self.early_stopping_metric]

        # set scheduler
        if self.scheduler_step_at == "validation":
            self.scheduler.step(metrics=ckpt_score)
        
        # update new best
        new_best = self.stats.is_best(ckpt_score)
        if new_best:
            self.stats.best_ckpt_score = ckpt_score
            self.stats.best_ckpt_iter = self.stats.steps
            logger.info("Hooray! New best validation score [%s]!", self.early_stopping_metric)

        # save checkpoints
        is_better = self.stats.is_better(ckpt_score, self.ckpt_queue)
        if is_better:
            self.save_model_checkpoint(new_best, ckpt_score)
        
        # append to validation report 
        self.add_validation_report(valid_scores=valid_scores, new_best=new_best)
        self.log_examples(valid_hypotheses, valid_references,
                          valid_hypotheses_raw, data=valid_data)

        # store validation set outputs
        validate_output_path = Path(self.model_dir) / f"{self.stats.steps}.hyps"
        write_validation_output_to_file(validate_output_path, valid_hypotheses)

        # store attention plot for selected valid sentences
        # TODO

        return valid_duration_time
    
    def add_validation_report(self, valid_scores:dict, new_best: bool) -> None:
        """
        Append a one-line report to validation logging file.
        """
        current_lr = self.optimizer.param_groups[0]["lr"]
        valid_file = Path(self.model_dir) / "validation.log"
        with valid_file.open("a", encoding="utf-8") as fg:
            score_string = "\t".join([f"Steps: {self.stats.steps}"] + 
            [f"{eval_metric}: {score:.5f}" for eval_metric, score in valid_scores.items()] +
            [f"LR: {current_lr:.8f}", "*" if new_best else ""])
            fg.write(f"{score_string}\n") 

        return None

    def log_examples(self, hypotheses:List[str], references:List[str],
                     hypotheses_raw: List[List[str]], data:Dataset) -> None:
        """
        Log the first self.log_valid_sentences from given examples.
        hypotheses: decoded hypotheses (list of strings)
        hypotheses_raw: raw hypotheses (list of list of tokens)
        references: decoded references (list of strings)
        """
        for id in self.log_valid_sentences:
            if id >= len(hypotheses):
                continue
            logger.info("Example #%d", id)

            # tokenized text
            tokenized_src = data.get_item(idx=id, lang=data.src_lang)
            tokenized_trg = data.get_item(idx=id, lang=data.trg_lang)
            logger.debug("\tTokenized source:  %s", tokenized_src)
            logger.debug("\tTokenized reference:  %s", tokenized_trg)
            logger.debug("\tTokenized hypothesis:  %s", hypotheses_raw[id])
            # FIXME what is tokenized and what is detokenized.
            # detokenized text
            logger.info("\tSource:  %s",data.src[id])
            logger.info("\tReference:  %s", references[id])
            logger.info("\tHypothesis: %s", hypotheses[id])

        return None

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