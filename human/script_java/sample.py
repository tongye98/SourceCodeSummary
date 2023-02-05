import numpy as np
sample_number = np.random.choice(8704, 100, replace=False)

with open("../../data/script_java/test.token.code","r") as fcode, open("../../data/script_java/test.token.nl","r") as ftruth:
    with open("java_default_beam.txt", "r") as script, open("base","r") as base, open("full", "r") as full:
        with open("sample_code","w") as sample_code, open("script_summary","w") as sample_script, open("base_summary","w") as sample_base, open("full_summary","w") as sample_full, open("ground_truth","w") as sample_truth:
            
            code = fcode.read().splitlines()
            summary = ftruth.read().splitlines()

            raw_script = script.read().splitlines()
            raw_base = base.read().splitlines()
            raw_full = full.read().splitlines()

            for number in sample_number:
                sample_code.write(code[number]+'\n')
                sample_truth.write(summary[number]+'\n')

                sample_script.write(raw_script[number]+'\n')
                sample_base.write(raw_base[number]+'\n')
                sample_full.write(raw_full[number]+'\n')


