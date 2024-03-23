# TRAM
This repo serves as the official implementation of NAACL 2024 findings paper "[Tram: A Token-level Retrieval-augmented Mechanism for Source Code Summarization](https://arxiv.org/abs/2305.11074)".

We propose a token-level retrieval-augmented mechanism（Tram）to generate a better source code summary.

If you have any questions, be free to email me.

## Abstract
Automatically generating human-readable text describing the functionality of a program is the intent of source code summarization. Although neural language models achieve significant performance in this field, they are limited by their inability to access external knowledge. To address this limitation, an emerging trend is combining neural models with external knowledge through retrieval methods. Previous methods have relied on the sentence-level retrieval paradigm on the encoder side. However, this paradigm is coarse-grained, noise-filled and cannot directly take advantage of the high-quality retrieved summary tokens on the decoder side. In this paper, we propose a fine-grained Token-level retrieval-augmented mechanism (Tram) on the decoder side rather than the encoder side to enhance the performance of neural models and produce more low-frequency tokens in generating summaries. Furthermore, to overcome the challenge of token-level retrieval in capturing contextual code semantics, we also propose integrating code semantics into individual summary tokens. The results of extensive experiments and human evaluation show that our token-level retrieval-augmented approach significantly improves performance and is more interpretable.

## Architecture

![datas](../figs/architecure.pdf)
![datas2](../figs/image.png)


## Dependency

```bash
pip install -r requirements.txt
```

## Quick Start
**All training, build datastore, retrieval and model parameters are in the [config](configs/): yaml file.**


**Step 1: Training**

```bash
export CUDA_VISIBLE_DEVICES=1
python -m src train configs/codescribe_python.yaml
```

**Step 2: Testing**
```bash
python -m src test configs/codescribe_python.yaml --ckpt models/codescribe_python/best.ckpt
```

**Step 3: Build Datastore**
```bash
python -m src build_database configs/codescribe_python.yaml
```

**Step 4: Retrieval-based Generation**
```bash
python -m src retrieval_test configs/codescribe_python.yaml  --ckpt  models/codescribe_python/best.ckpt
```


## Citation
```
@article{ye2023tram,
  title={Tram: A Token-level Retrieval-augmented Mechanism for Source Code Summarization},
  author={Ye, Tong and Wu, Lingfei and Ma, Tengfei and Zhang, Xuhong and Du, Yangkai and Liu, Peiyu and Wang, Wenhai and Ji, Shouling},
  journal={arXiv preprint arXiv:2305.11074},
  year={2023}
}
```