import numpy as np
sample_number = np.random.choice(21745, 100, replace=False)

with open("test.code","r") as fcode, open("test.summary","r") as ftruth:
    with open("rencos.out", "r") as rencos, open("output_beam4.test","r") as base, open("output_analysis_beam4_mx=0.5bandwidth=20topk=8", "r") as full:
        with open("sample_code","w") as sample_code, open("sample_rencos","w") as sample_rencos, open("sample_base","w") as sample_base, open("sample_full","w") as sample_full, open("samplt_truth","w") as sample_truth:
            
            code = fcode.read().splitlines()
            summary = ftruth.read().splitlines()

            raw_rencos = rencos.read().splitlines()
            raw_base = base.read().splitlines()
            raw_full = full.read().splitlines()

            for number in sample_number:
                sample_code.write(code[number]+'\n')
                sample_truth.write(summary[number]+'\n')

                sample_rencos.write(raw_rencos[number]+'\n')
                sample_base.write(raw_base[number]+'\n')
                sample_full.write(raw_full[number]+'\n')


