"""
Rencos Test in Transformer.
"""
import logging 
import torch 
from pathlib import Path
from src.datas import load_data
from src.model import build_model
from src.rencos_prediction import predict
from src.helps import load_config, make_logger
from src.helps import load_model_checkpoint, write_list_to_file

logger = logging.getLogger(__name__)

def rencos_test(cfg_file: str, ckpt_path:str=None) -> None:
    """
    Main rencos test function. Handles loading a model from checkpoint, generating translation.
    :param cfg_file: path to configuration file
    :param ckpt: path to model checkpoint to load
    :param datasets: dict, to predict
    """ 
    cfg = load_config(Path(cfg_file))
    model_dir = cfg["training"].get("model_dir", None)
    assert model_dir is not None 

    # make logger
    make_logger(Path(model_dir), mode="rencos_beam4_oursimilar3_lambda1_sameword")

    use_cuda = cfg["training"].get("use_cuda", False) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    num_workers = cfg["training"].get("num_workers", 0)
    normalization = cfg["training"].get("normalization","batch")

    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(data_cfg=cfg["data"], use_rencos=True)
    data_to_predict = {"test":test_data}

    # build an transformer(encoder-decoder) model
    model = build_model(model_cfg=cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # load model checkpoint
    logger.info("ckpt_path = {}".format(ckpt_path))
    model_checkpoint = load_model_checkpoint(path=Path(ckpt_path), device=device)

    #restore model and optimizer parameters
    model.load_state_dict(model_checkpoint["model_state"])
    
    if device.type == "cuda":
        model.to(device)

    # Test
    for dataset_name, dataset in data_to_predict.items():
        if dataset is not None:
            logger.info("Testing on %s set...", dataset_name)
            (valid_scores, valid_references, valid_hypotheses, valid_sentences_scores, 
            valid_attention_scores) = predict(model=model, data=dataset, device=device, compute_loss=False, 
            normalization=normalization, num_workers=num_workers, test_cfg=cfg["testing"])

            for eval_metric, score in valid_scores.items():
                if eval_metric in ["loss", "ppl"]:
                    logger.info("eval metric {} = {}".format(eval_metric, score))
                else:
                    logger.info("eval metric {} = {}".format(eval_metric, score*100))

            if valid_hypotheses is not None:
                # save final model outputs.
                test_output_path = Path(model_dir) / "output_rencos_beam4_oursimilar3_lamda1_sameword_test.{}".format(dataset_name)
                write_list_to_file(file_path=test_output_path, array=valid_hypotheses)
                logger.info("Results saved to: %s.", test_output_path)
        else:
            logger.info(f"{dataset_name} is not exist!" )
            