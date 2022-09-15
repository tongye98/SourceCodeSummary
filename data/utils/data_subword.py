import sentencepiece as spm

# spm.SentencePieceTrainer.train(input="data/utils/rencos_python_full.code", model_prefix="data/utils/rencos_python_sub", vocab_size=50000)

sp = spm.SentencePieceProcessor(model_file="data/utils/rencos_python_sub.model")

with open("data/rencos_python/train.code",'r') as ftrain, open("data/rencos_python_sub/train.code", 'w') as gtrain:
    lines = ftrain.read().splitlines()
    for line in lines:
        line = sp.EncodeAsPieces(input=line)
        line = " ".join(line)
        line = line.replace("▁", "")
        gtrain.write(line+'\n')

with open("data/rencos_python/valid.code",'r') as fvalid, open("data/rencos_python_sub/valid.code", 'w') as gvalid:
    lines = fvalid.read().splitlines()
    for line in lines:
        line = sp.EncodeAsPieces(input=line)
        line = " ".join(line)
        line = line.replace("▁", "")
        gvalid.write(line+'\n')

with open("data/rencos_python/test.code",'r') as ftest, open("data/rencos_python_sub/test.code", 'w') as gtest:
    lines = ftest.read().splitlines()
    for line in lines:
        line = sp.EncodeAsPieces(input=line)
        line = " ".join(line)
        line = line.replace("▁", "")
        gtest.write(line+'\n')
        






