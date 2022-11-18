import json
from utils import config
attr_dict_path = config['attr_dict_path'].replace('.pt', '.json')
attr_dict = json.load(open(attr_dict_path, 'r', encoding='utf8'))
# count number of lines per character
attr_dict_lens = {}
for txt_path, txt_dict in attr_dict.items():
    for character, line_numbers in txt_dict.items():
        if character not in attr_dict_lens:
            attr_dict_lens[character] = 0
        attr_dict_lens[character] += len(line_numbers)

# print data
for character, line_count in sorted(attr_dict_lens.items(), key=lambda x: x[1], reverse=True):
    if line_count > 100:
        print(character, line_count)
print('Total characters:', len(attr_dict_lens))
print('Total characters (over  20):', len([x for x in attr_dict_lens.values() if x >  20]))
print('Total characters (over  50):', len([x for x in attr_dict_lens.values() if x >  50]))
print('Total characters (over 100):', len([x for x in attr_dict_lens.values() if x > 100]))
print('Total lines:', sum(attr_dict_lens.values()))
print('Total lines (over  20):', sum([x for x in attr_dict_lens.values() if x >  20]))
print('Total lines (over  50):', sum([x for x in attr_dict_lens.values() if x >  50]))
print('Total lines (over 100):', sum([x for x in attr_dict_lens.values() if x > 100]))