import logging 
import torch 
from pathlib import Path
from src.datas import load_data
from src.model import build_model
from src.retrieval_prediction import predict
from src.helps import load_config, make_logger
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

    model_dir = cfg["retriever"].get("retriever_model_dir", None)
    use_code_representation = cfg["retriever"]["use_code_representation"]
    assert model_dir is not None 

    # make logger
    make_logger(Path(model_dir), mode="retriever_beam4_l2_nocodesemantic")

    use_cuda = cfg["training"].get("use_cuda", False) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    num_workers = cfg["training"].get("num_workers", 0)
    normalization = cfg["training"].get("normalization","batch")

    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(data_cfg=cfg["data"])
    data_to_predict = {"dev": dev_data, "test":test_data}

    # build an transformer(encoder-decoder) model
    model = build_model(model_cfg=cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # load model checkpoint
    logger.info("ckpt_path = {}".format(ckpt_path))
    model_checkpoint = load_model_checkpoint(path=Path(ckpt_path), device=device)

    #restore model and optimizer parameters
    model.load_state_dict(model_checkpoint["model_state"])

    retriever = build_retriever(retriever_cfg=cfg["retriever"])
    model.retriever =  retriever 
    
    if device.type == "cuda":
        model.to(device)

    # grid search 
    for mixing_weight in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1]:
        for bandwidth in [10, 20, 50, 1, 100]:
            for top_k in [4, 8, 16]:
                logger.info("mixing_weight = {} | bandwidth = {} | top_k = {}".format(mixing_weight, bandwidth, top_k))

                model.retriever.mixing_weight = mixing_weight
                model.retriever.bandwidth = float(bandwidth)
                model.retriever.top_k = top_k

                # Test
                for dataset_name, dataset in data_to_predict.items():
                    if dataset_name != "test":
                        continue
                    if dataset is not None:
                        logger.info("Testing on %s set...", dataset_name)
                        (valid_scores, valid_references, valid_hypotheses, valid_sentences_scores, 
                        valid_attention_scores) = predict(model=model, data=dataset, device=device, compute_loss=False, 
                        normalization=normalization, num_workers=num_workers, test_cfg=cfg["testing"],
                        use_code_representation=use_code_representation)

                        if valid_hypotheses is not None:
                            # save final model outputs.
                            test_output_path = Path(model_dir) / "output_beam4_l2_mx={}bandwidth={}topk={}_nocodesemantic".format(mixing_weight, bandwidth, top_k)
                            write_list_to_file(file_path=test_output_path, array=valid_hypotheses)
                            logger.info("Results saved to: %s.", test_output_path)
                    else:
                        logger.info(f"{dataset_name} is not exist!" )