from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
tokenizer.save_vocabulary('data') # creates 'data/vocab.txt'

from utils import config, strip_to_word
import json
attr_dict_path = config['attr_dict_path'].replace('.pt', '.json')
attr_dict = json.load(open(attr_dict_path, 'r', encoding='utf8'))
# get characters sorted by number of lines (descending)
attr_dict_lens = {}
for txt_path, txt_dict in attr_dict.items():
    for character_ws, line_numbers in txt_dict.items():
        if character_ws not in attr_dict_lens:
            attr_dict_lens[character_ws] = 0
        attr_dict_lens[character_ws] += len(line_numbers)
characters = [strip_to_word(x[0]) for x in sorted(attr_dict_lens.items(), key=lambda x: x[1], reverse=True)]

# add characters to vocab
existing_vocab = open('data/vocab.txt', 'r', encoding='utf8').read().splitlines()
new_vocab = existing_vocab + [c for c in characters if c not in existing_vocab]
new_vocab[new_vocab.index('[MASK]')] = 'MASKTOKEN' # hack since tokenizer can't handle '[MASK].' -> [103, 119]
with open('data/vocab.txt', 'w', encoding='utf8') as f:
    f.write("\n".join(new_vocab))