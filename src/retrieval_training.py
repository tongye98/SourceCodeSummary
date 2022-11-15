import logging
import numpy as np
import torch
import heapq
import math
import time
import shutil
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
from src.helps import load_config, make_model_dir, make_logger, log_cfg
from src.helps import set_seed, parse_train_arguments, load_model_checkpoint
from src.helps import symlink_update, delete_ckpt, write_validation_output_to_file
from src.prediction import predict
from src.datas import load_data, make_data_iter
from src.model import build_model, Transformer
from torch.utils.tensorboard import SummaryWriter
from src.builders import build_gradient_clipper, build_optimizer, build_scheduler
from src.retriever import build_retriever

logger = logging.getLogger(__name__) 

def retrieval_train(cfg_file: str) -> None:
    """
    Retrieval Training function. 
    """
    cfg = load_config(Path(cfg_file))

    # make model dir
    model_dir = cfg["retriever"].get("retrieval_model_dir", None)

    # make logger
    make_logger(model_dir, mode="retrieval_train")

    # write all entries of config to the log file.
    log_cfg(cfg)

    # set the whole random seed
    set_seed(seed=int(cfg["training"].get("random_seed", 980820)))

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(data_cfg=cfg["data"])

    # load model state from trained model
    pre_trained_model_path = cfg["retriever"].get("pre_trained_model_path", None)
    use_cuda = cfg["training"].get("use_cuda", False)
    device = torch.device("cuda" if use_cuda else "cpu")
    model_checkpoint = load_model_checkpoint(path=Path(pre_trained_model_path), device=device)

    # build an transformer(encoder-decoder) model
    model = build_model(model_cfg=cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    for p in model.parameters():
        p.requires_grad = False

    retriever = build_retriever(retriever_cfg=cfg["retriever"])
    # load combiner from checkpoint for dynamic combiners

    model.retriever = retriever
    model.log_parameters_list()

    # for training management.
    trainer = RetrievalTrainManager(model=model, cfg=cfg)

    # train the model
    trainer.train_and_validate(train_data=dev_data, valid_data=test_data)

class RetrievalTrainManager(object):
    """
    Retrieval train manager.
    Manage the whole training loop, valiation, learning rate scheduling and early stopping.
    """
    def __init__(self, model: Transformer, cfg: dict) -> None:

        (model_dir, loss_function, label_smoothing,
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
        self.tb_writer = SummaryWriter(log_dir=(model_dir/"retrieval_tensorboard_log").as_posix())

        # model
        self.model = model

        # CPU/GPU
        self.device = device
        self.n_gpu = n_gpu
        self.num_workers = num_workers
        if self.device.type == "cuda":
            self.model.to(self.device)
        
        # optimization
        self.clip_grad_fun = build_gradient_clipper(train_cfg=cfg["training"])
        self.optimizer = build_optimizer(train_cfg=cfg["training"], parameters=self.model.retriever.parameters())

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
        self.scheduler, self.scheduler_step_at = build_scheduler(train_cfg=cfg["training"], optimizer=self.optimizer)

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
            steps=0, is_min_lr=False, is_max_update=False,
            total_tokens=0, best_ckpt_iter=0, minimize_metric = self.minimize_metric,
            best_ckpt_score=np.inf if self.minimize_metric else -np.inf,
        )
        
        if load_model is not None:
            self.init_from_checkpoint(load_model,
                reset_best_ckpt=reset_best_ckpt, reset_scheduler=reset_scheduler,
                reset_optimizer=reset_optimizer, reset_iter_state=reset_iter_state)

        # config for generation
        self.valid_cfg = cfg["testing"].copy()
        self.valid_cfg["beam_size"] = 1 # 1 means greedy decoding during train-loop validation
        self.valid_cfg["batch_size"] = self.batch_size * 2
        self.valid_cfg["batch_type"] = self.batch_type
        self.valid_cfg["n_best"] = 1   # only the best one
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
        self.train_iter = make_data_iter(dataset=train_data, sampler_seed=self.seed, shuffle=self.shuffle, batch_type=self.batch_type,
                                         batch_size=self.batch_size, num_workers=self.num_workers)
        
        # train and validate main loop
        logger.info("Train stats:\n"
                    "\tdevice: %s\n"
                    "\tn_gpu: %d\n"
                    "\tbatch_size: %d",
                    self.device.type, self.n_gpu, self.batch_size)
        try:
            self.model.eval()
            for epoch_no in range(self.epochs):
                logger.info("Epoch %d", epoch_no + 1)
                    
                self.model.retriever.train()
                self.model.retriever.zero_grad()

                # Statistic for each epoch.
                start_time = time.time()
                start_tokens = self.stats.total_tokens
                epoch_loss = 0

                for batch_data in self.train_iter:
                    batch_data.move2cuda(self.device)
                    normalized_batch_loss = self.train_step(batch_data)
                    
                    normalized_batch_loss.backward()
                        
                    # clip gradients (in-place)
                    if self.clip_grad_fun is not None:
                        self.clip_grad_fun(parameters=self.model.retriever.parameters())
                    
                    # make gradient step
                    self.optimizer.step()

                    # logging gradient information 
                    for (name, param) in self.model.retriever.named_parameters():
                        logger.debug("Name = {}, Param={}".format(name, param))
                        logger.debug("Gradient {}".format(param.grad))

                    # reset gradients
                    self.model.retriever.zero_grad()

                    # accumulate loss
                    epoch_loss += normalized_batch_loss.item()

                    # increment token counter
                    self.stats.total_tokens += batch_data.ntokens
                    
                    # increment step counter
                    self.stats.steps += 1
                    if self.stats.steps >= self.max_updates:
                        self.stats.is_max_update == True
                    
                    # check current leraning rate(lr)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    if current_lr < self.learning_rate_min:
                        self.stats.is_min_lr = True 

                    # log learning process and write tensorboard
                    if self.stats.steps % self.logging_freq == 0:
                        elapse_time = time.time() - start_time
                        elapse_token_num = self.stats.total_tokens - start_tokens

                        self.tb_writer.add_scalar(tag="Train/batch_loss", scalar_value=normalized_batch_loss, global_step=self.stats.steps)
                        self.tb_writer.add_scalar(tag="Train/learning_rate", scalar_value=current_lr, global_step=self.stats.steps)

                        logger.info("Epoch %3d, Step: %7d, Batch Loss: %12.6f, Lr: %.6f, Tokens per sec: %6.0f",
                        epoch_no + 1, self.stats.steps, normalized_batch_loss, self.optimizer.param_groups[0]["lr"],
                        elapse_token_num / elapse_time)

                        start_time = time.time()
                        start_tokens = self.stats.total_tokens
                    
                    # decay learning_rate(lr)
                    if self.scheduler_step_at == "step":
                        self.scheduler.step(self.stats.steps)

                logger.info("Epoch %3d: total training loss %.2f", epoch_no + 1, epoch_loss)

                if self.scheduler_step_at == "epoch":
                    self.scheduler.step()

                # validate on the entire dev dataset
                if (epoch_no + 1) % self.validation_freq == 0:
                    valid_duration_time = self.validate(valid_data)
                    logger.info("Validation time = {}s.".format(valid_duration_time))

                # check after a number of whole epoch.
                if self.stats.is_min_lr or self.stats.is_max_update:
                    log_string = (f"minimum learning rate {self.learning_rate_min}" if self.stats.is_min_lr else 
                                    f"maximun number of updates(steps) {self.max_updates}")
                    logger.info("Training enede since %s was reached!", log_string)
                    break 
            else: # normal ended after training.
                logger.info("Training ended after %3d epoches!", epoch_no + 1)

            logger.info("Best Validation result (greedy) at step %8d: %6.2f %s.",
                        self.stats.best_ckpt_iter, self.stats.best_ckpt_score, self.early_stopping_metric)
                    
        except KeyboardInterrupt:
            self.save_model_checkpoint(False, float("nan"))

        self.tb_writer.close()
        return None 

    def train_step(self, batch_data):
        """
        Train the model on one batch: compute loss
        """
        src_input = batch_data.src
        trg_input = batch_data.trg_input
        src_mask = batch_data.src_mask
        trg_mask = batch_data.trg_mask
        trg_truth = batch_data.trg_truth

        # get loss (run as during training with teacher forcing)
        batch_loss = self.model(return_type="retrieval_loss", src_input=src_input, trg_input=trg_input,
                   src_mask=src_mask, trg_mask=trg_mask, encoder_output = None, trg_truth=trg_truth)

        # normalization = 'batch' means final loss is average-sentence level loss in batch
        # normalization = 'tokens' means final loss is average-token level loss in batch
        normalized_batch_loss = batch_data.normalize(batch_loss, "sum")
        return normalized_batch_loss

    def validate(self, valid_data: Dataset):
        """
        Validate on the valid dataset.
        return the validate time.
        """
        validate_start_time = time.time()
        # vallid_hypotheses_raw is befor tokenizer post_process
        (valid_scores, valid_references, valid_hypotheses, 
        valid_sentences_scores, valid_attention_scores) = predict(model=self.model, data=valid_data, device=self.device, 
             compute_loss=True, normalization=self.normalization, num_workers=self.num_workers, test_cfg=self.valid_cfg)
        valid_duration_time = time.time() - validate_start_time
        
        # write eval_metric and corresponding score to tensorboard
        for eval_metric, score in valid_scores.items():
            # if not math.isnan(score):
            if eval_metric in ["loss", "ppl"]:
                self.tb_writer.add_scalar(tag=f"Valid/{eval_metric}", scalar_value=score, global_step=self.stats.steps)
            else:
                self.tb_writer.add_scalar(tag=f"Valid/{eval_metric}", scalar_value=score*100, global_step=self.stats.steps)
        
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
        is_better = self.stats.is_better(ckpt_score, self.ckpt_queue) if len(self.ckpt_queue) > 0 else True
        if is_better or self.num_ckpts < 0:
            self.save_model_checkpoint(new_best, ckpt_score)
        
        # append to validation report 
        self.add_validation_report(valid_scores=valid_scores, new_best=new_best)
        self.log_examples(valid_hypotheses, valid_references, data=valid_data)

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
            score_string = "\t".join([f"Steps: {self.stats.steps:7}"] + 
            [f"{eval_metric}: {score*100:.2f}" if eval_metric in ["bleu", "meteor", "rouge-l"] 
            else f"{eval_metric}: {score:.2f}" for eval_metric, score in valid_scores.items()] +
            [f"LR: {current_lr:.8f}", "*" if new_best else ""])
            fg.write(f"{score_string}\n") 

    def log_examples(self, hypotheses:List[str], references:List[str], data:Dataset) -> None:
        """
        Log the first self.log_valid_sentences from given examples.
        hypotheses: decoded hypotheses (list of strings)
        references: decoded references (list of strings)
        """
        for id in self.log_valid_sentences:
            if id >= len(hypotheses):
                continue
            logger.info("Example #%d", id)

            # detokenized text
            logger.info("\tSource:  %s", data.original_data[data.src_language][id])
            logger.info("\tReference:  %s", references[id])
            logger.info("\tHypothesis: %s", hypotheses[id])

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
