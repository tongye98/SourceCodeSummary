# merge the train valid test code to a same file to sentencepiece

with open("data/utils/rencos_python_full.code", 'w') as ff:
    with open("data/rencos_python/train.code", 'r') as ftrain:
        lines = ftrain.readlines()
        for line in lines:
            ff.write(line)
    with open("data/rencos_python/valid.code", 'r') as fvalid:
        lines = fvalid.readlines()
        for line in lines:
            ff.write(line)
    with open("data/rencos_python/test.code",'r') as ftest:
        lines = ftest.readlines()
        for line in lines:
            ff.write(line)