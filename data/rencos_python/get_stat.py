import subprocess
from tqdm import tqdm
from prettytable import PrettyTable


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').split(' ')
    return int(num[0])


def main():
    records = {'train': 0, 'valid': 0, 'test': 0}
    function_tokens = {'train': 0, 'valid': 0, 'test': 0}
    summary_tokens = {'train': 0, 'valid': 0, 'test': 0}
    unique_function_tokens = {'train': set(), 'valid': set(), 'test': set()}
    unique_summary_tokens = {'train': set(), 'valid': set(), 'test': set()}

    attribute_list = ["Records", "Function Tokens", "Summary Tokens",
                      "Unique Function Tokens", "Unique Summary Tokens"]

    def read_data(split):
        source = '%s.code' % split
        target = '%s.summary' % split
        with open(source) as f1, open(target) as f2:
            for src, tgt in tqdm(zip(f1, f2),
                                 total=count_file_lines(source)):
                func_tokens = src.strip().split()
                summ_tokens = tgt.strip().split()
                records[split] += 1
                function_tokens[split] += len(func_tokens)
                summary_tokens[split] += len(summ_tokens)
                unique_function_tokens[split].update(func_tokens)
                unique_summary_tokens[split].update(summ_tokens)

    read_data('train')
    read_data('valid')
    read_data('test')

    table = PrettyTable()
    table.field_names = ["Attribute", "Train", "Valid", "Test", "Fullset"]
    table.align["Attribute"] = "l"
    table.align["Train"] = "r"
    table.align["Valid"] = "r"
    table.align["Test"] = "r"
    table.align["Fullset"] = "r"
    for attr in attribute_list:
        var = eval('_'.join(attr.lower().split()))
        val1 = len(var['train']) if isinstance(var['train'], set) else var['train']
        val2 = len(var['valid']) if isinstance(var['valid'], set) else var['valid']
        val3 = len(var['test']) if isinstance(var['test'], set) else var['test']
        fullset = val1 + val2 + val3
        table.add_row([attr, val1, val2, val3, fullset])

    avg = (function_tokens['train'] + function_tokens['valid'] + function_tokens['test']) / (
            records['train'] + records['valid'] + records['test'])
    table.add_row([
        'Avg. Function Length',
        '%.2f' % (function_tokens['train'] / records['train']),
        '%.2f' % (function_tokens['valid'] / records['valid']),
        '%.2f' % (function_tokens['test'] / records['test']),
        '%.2f' % avg
    ])
    avg = (summary_tokens['train'] + summary_tokens['valid'] + summary_tokens['test']) / (
            records['train'] + records['valid'] + records['test'])
    table.add_row([
        'Avg. Summary Length',
        '%.2f' % (summary_tokens['train'] / records['train']),
        '%.2f' % (summary_tokens['valid'] / records['valid']),
        '%.2f' % (summary_tokens['test'] / records['test']),
        '%.2f' % avg
    ])
    print(table)


if __name__ == '__main__':
    main()
