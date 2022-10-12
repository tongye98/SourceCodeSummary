import logging 
import torch 
from pathlib import Path
from src.datas import load_data
from src.model import build_model
from src.retrieval_prediction import predict
from src.helps import load_config, make_logger, resolve_ckpt_path
from src.helps import load_model_checkpoint, write_list_to_file
from src.retriever import build_retriever

logger = logging.getLogger(__name__)

def retrieval_test(cfg_file: str, ckpt_path:str=None) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating translation.
    :param cfg_file: path to configuration file
    :param ckpt: path to model checkpoint to load
    :param datasets: dict, to predict
    """ 
    cfg = load_config(Path(cfg_file))

    model_dir = cfg["retrieval_training"].get("model_dir", None)
    assert model_dir is not None 

    # make logger
    make_logger(Path(model_dir), mode="retrieval_test")

    load_model = cfg["retrieval_training"].get("load_model", None)
    use_cuda = cfg["retrieval_training"].get("use_cuda", False) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    num_workers = cfg["retrieval_training"].get("num_workers", 0)
    normalization = cfg["retrieval_training"].get("normalization","batch")

    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(data_cfg=cfg["data"])
    data_to_predict = {"dev": dev_data, "test":test_data}

    # build an transformer(encoder-decoder) model
    model = build_model(model_cfg=cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # when checkpoint is not specified, take latest(best) from model dir
    ckpt_path = cfg["retrieval_training"].get("pre_trained_model_path", None)
    logger.info("ckpt_path = {}".format(ckpt_path))

    # load model checkpoint
    model_checkpoint = load_model_checkpoint(path=Path(ckpt_path), device=device)

    #restore model and optimizer parameters
    model.load_state_dict(model_checkpoint["model_state"])

    retriever_type = cfg["retriever"]["type"]

    retriever = build_retriever(retriever_cfg=cfg["retriever"])
    model.retriever =  retriever 
    
    if device.type == "cuda":
        model.to(device)

    # grid search 
    for mixing_weight in [0.5, 0.6, 0.7]:
        for bandwidth in [20, 25, 30]:
            for top_k in [8, 16, 32]:
                logger.info("mixing_weight = {} | bandwidth = {} | top_k = {}".format(mixing_weight, bandwidth, top_k))

                model.retriever.mixing_weight = mixing_weight
                model.retriever.bandwidth = float(bandwidth)
                model.retriever.top_k = top_k

                # Test
                for dataset_name, dataset in data_to_predict.items():
                    if dataset_name == "dev":
                        continue
                    if dataset is not None:
                        logger.info("Testing on %s set...", dataset_name)
                        (valid_scores, valid_references, valid_hypotheses, valid_sentences_scores, 
                        valid_attention_scores) = predict(model=model, data=dataset, device=device, compute_loss=True, 
                        normalization=normalization, num_workers=num_workers, test_cfg=cfg["testing"])
                        for eval_metric, score in valid_scores.items():
                            if eval_metric in ["loss", "ppl"]:
                                logger.info("eval metric {} = {}".format(eval_metric, score))
                            else:
                                logger.info("eval metric {} = {}".format(eval_metric, score*100))
                        if valid_hypotheses is not None:
                            # save final model outputs.
                            test_output_path = Path(model_dir) / "output_{}_mw_{}_band_{}_topk_{}.{}".format(retriever_type, mixing_weight, bandwidth, top_k, dataset_name)
                            write_list_to_file(file_path=test_output_path, array=valid_hypotheses)
                            logger.info("Results saved to: %s.", test_output_path)
                    else:
                        logger.info(f"{dataset_name} is not exist!" )