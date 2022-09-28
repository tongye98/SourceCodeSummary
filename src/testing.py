def test(cfg_file: str, ckpt_path:str, output_path:str=None, datasets:dict=None, save_scores:bool=True, save_attention:bool=False) -> None:
    """
    Main test function.
    Handles loading a model from checkpoint, generating translation, storing, and (optional) plotting attention.
    :param cfg_file: path to configuration file
    :param ckpt: path to model checkpoint to load
    :param output_path: path to output
    :param datasets: dict, to predict
    :param: sava_scores: whether to save scores
    :param: save_attention: whether to save attention visualization
    """ 
    cfg = load_config(Path(cfg_file))

    model_dir = cfg["training"].get("model_dir", None)
    load_model = cfg["training"].get("load_model", None)
    use_cuda = cfg["training"].get("use_cuda", False) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    n_gpu = torch.cuda.device_count() if use_cuda else 0
    num_workers = cfg["training"].get("num_workers", 0)
    normalization = cfg["training"].get("normalization","batch")
    seed = cfg["training"].get("random_seed", 980820)

    if datasets is None:
        train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(data_cfg=cfg["data"])
        data_to_predict = {"dev": dev_data, "test":test_data}
    else:
        data_to_predict = {"dev":datasets["dev"], "test":datasets["test"]}
        src_vocab = datasets["src_vocab"]
        trg_vocab = datasets["trg_vocab"]
    
    # build an transformer(encoder-decoder) model
    model = build_model(model_cfg=cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    # model.log_parameters_list()

    # when checkpoint is not specified, take latest(best) from model dir
    ckpt_path = resolve_ckpt_path(ckpt_path, load_model, model_dir)

    # load model checkpoint
    model_checkpoint = load_model_checkpoint(path=ckpt_path, device=device)

    #restore model and optimizer parameters
    model.load_state_dict(model_checkpoint["model_state"])
    if device.type == "cuda":
        model.to(device)

    # really test
    for dataset_name, dataset in data_to_predict.items():
        if dataset is not None:
            logger.info("Testing on %s set...", dataset_name)
            # FIXME compute_loss is set to true
            (valid_scores, bleu_order, valid_references, valid_hypotheses, decoded_valid,
            valid_sentences_scores, valid_attention_scores) = predict(model=model, data=dataset, device=device, n_gpu=n_gpu,
            compute_loss=True, normalization=normalization, num_workers=num_workers, cfg=cfg["testing"], seed=seed)
            if valid_hypotheses is not None:
                # save final model outputs.
                test_output_path = Path(f"{output_path}.{dataset_name}")
                write_list_to_file(file_path=test_output_path, array=valid_hypotheses)
                logger.info("Results saved to: %s.", test_output_path)
        else:
            logger.info(f"{dataset_name} is not exist!" )