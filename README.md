# TRAM
This repo serves as the official implementation of NAACL 2024 findings paper "[Tram: A Token-level Retrieval-augmented Mechanism for Source Code Summarization](https://arxiv.org/abs/2305.11074)".

We propose a token-level retrieval-augmented mechanism（Tram）to generate a better source code summary.

If you have any questions, be free to email me.

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