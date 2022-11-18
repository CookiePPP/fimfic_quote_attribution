# Adapted from https://github.com/yang-zhang/lightning-language-modeling/blob/main/data.py
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
import json
import os
import traceback
import warnings
import pytorch_lightning as pl
import torch
import random
from torch.utils.data.dataloader import DataLoader
from unidecode import unidecode

from patch_model import get_patched_distilbert
from utils import sample_from_dict, process_character, config

def maybe_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def split_attr_dict(attr_dict, p_val=0.005):
    """Split the attribute dictionary into train and test sets."""
    import random
    random.seed(0)
    train_dict = {}
    test_dict = {}
    for txt_path, txt_dict in attr_dict.items():
        train_dict[txt_path] = {}
        test_dict[txt_path] = {}
        for character_ws, line_numbers in txt_dict.items():
            if character_ws not in train_dict[txt_path]:
                train_dict[txt_path][character_ws] = []
            if character_ws not in test_dict[txt_path]:
                test_dict[txt_path][character_ws] = []
            for line_number in line_numbers:
                if random.random() < p_val:
                    test_dict[txt_path][character_ws].append(line_number)
                else:
                    train_dict[txt_path][character_ws].append(line_number)
            if not train_dict[txt_path][character_ws]:
                del train_dict[txt_path][character_ws]
            if not test_dict[txt_path][character_ws]:
                del test_dict[txt_path][character_ws]
        if not train_dict[txt_path]:
            del train_dict[txt_path]
        if not test_dict[txt_path]:
            del test_dict[txt_path]
    return train_dict, test_dict

def simplify_line(line, attr_dict, line_num, random_obj=random):
    """If the line is already in the attribute dictionary, return the simplified version."""
    for character_ws in attr_dict:
        if line_num in attr_dict[character_ws]:
            # "{...} "Hi" {X} {verb} {...}" -> "Hi" X said.
            
            character = process_character(character_ws)
            line_split = line.split('"')
            j = random_obj.choice(list(range(len(line_split)))[2::2])
            line_split[j] = f' {character} said.' + line_split[j]
            line = '"'.join(line_split)
            return line
    return line

class CustomMLMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_seq_length: int, eval=False,
            simplify_line_chance=0.3, token_dropout=0.0):
        attr_dict = json.load(open(config['attr_dict_path'].replace('.pt', '.json'), 'r', encoding='utf-8'))
        train_dict, eval_dict = split_attr_dict(attr_dict)
        self.attr_dict = eval_dict if eval else train_dict
        self.attr_lens = {k: sum([len(v) for v in v.values()]) for k, v in self.attr_dict.items()}
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.eval = eval
        self.len = sum(self.attr_lens.values())
        self.simplify_line_chance = simplify_line_chance
        self.token_dropout = token_dropout
        
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id
    
    def process_text(self, text, attr_dict):
        text = "\n".join([unidecode(line).strip() for line in text.splitlines()])
        text = text.replace('  ', ' ')
        for character_ws in sorted(list(attr_dict.keys()), key=lambda x: len(x), reverse=True):
            character = process_character(character_ws)
            if character_ws != character:
                text = text.replace(character_ws, character)
        return text
    
    def get_text_from_path(self, txt_path, attr_dict, use_cache=False):
        maybe_remove(txt_path.replace('.txt', '.cache2'))
        maybe_remove(txt_path.replace('.txt', '.cache3'))
        
        text = None
        
        cache_path = txt_path.replace('.txt', '.cache4')
        if text is None and use_cache and os.path.exists(cache_path):
            text = open(cache_path, 'r', encoding='utf-8').read()
        
        if text is None:
            text = open(txt_path, 'r', encoding='utf-8').read()
            text = self.process_text(text, attr_dict)
            if use_cache:
                open(cache_path, 'w', encoding='utf-8').write(text)
        
        return text
    
    def _get_item(self, idx):
        random_obj = random.Random(idx) if self.eval else random.Random(random.random()+idx)
        
        # pick a random "said X" line from the dataset
        txt_path, attr_dict = sample_from_dict(self.attr_dict, self.attr_lens, random=random_obj)
        
        # load the text for this sample
        text = self.get_text_from_path(txt_path, attr_dict, use_cache=True)
        lines = text.splitlines()

        # randomly replace speech with known speaker with simplified format
        # (which may be used during inference for predicted samples)
        #for character_ws, line_num_list in attr_dict.items():
        #    character = process_character(character_ws)
        #    for line_num in line_num_list:
        #        if self.simplify_line_chance < random_obj.random():
        #            continue
        #        line = lines[line_num]
        #        # "{...} "Hi" {X} {verb} {...}" -> "Hi" X said.
        #        line_split = line.split('"')
        #        j = random_obj.choice(list(range(len(line_split)))[2::2])
        #        line_split[j] = f' {character} said.' + line_split[j]
        #        lines[line_num] = '"'.join(line_split)
        
        # load the specific line for this sample
        character_ws, line_num_list = sample_from_dict(attr_dict, random=random_obj)
        line_num = random_obj.choice(line_num_list)
        line = lines[line_num]
        if self.simplify_line_chance > random_obj.random():
            line = simplify_line(line, attr_dict, line_num, random_obj)
        character = process_character(character_ws)
        if character not in line:
            if character_ws not in line:
                raise AssertionError(f'character: "{character}" / "{character_ws}" is not in line below:\n{line}')
            line = line.replace(character_ws, character) # still not sure why how this happens
        
        text_ids = self.tokenizer(line)['input_ids'][1:-1] # list[int]
        character_ids = self.tokenizer(f' {character} ')['input_ids'][1:-1] # list[int] of length 1
        assert len(character_ids) == 1, f'character_ids: {character_ids}, character: {character}\nexpected length 1 but got length {len(character_ids)}'
        character_id = character_ids[0]
        n_character_ids = text_ids.count(character_id)
        assert n_character_ids > 0, f'character: "{character}" ({character_id}) is not in text ids produced from line below:\n{line}'
        
        # replace random character_id on selected line with [MASK]
        has_masked = False
        for i, text_id in (enumerate(text_ids) if n_character_ids == 1 else random.sample(list(enumerate(text_ids)), len(text_ids))):
            if text_id == character_id and (random_obj.random() < 0.5 or not has_masked):
                has_masked = True
                text_ids[i] = self.mask_id
        
        prev_lines = lines[:line_num]
        prev_lines_nums = list(range(line_num))
        next_lines = lines[line_num+1:]
        next_lines_nums = list(range(line_num+1, len(lines)))
        del lines
        
        # randomly add a line from the previous lines or the next lines till target length is reached
        while len(text_ids) < self.max_seq_length:
            if random_obj.random() < 0.5 and prev_lines:
                prev_line = prev_lines.pop(-1)
                prev_lines_num = prev_lines_nums.pop(-1)
                if self.simplify_line_chance > random_obj.random():
                    prev_line = simplify_line(prev_line, attr_dict, prev_lines_num, random_obj)
                text_ids = self.tokenizer(prev_line)['input_ids'][1:-1] + [self.sep_id] + text_ids
            elif next_lines:
                next_line = next_lines.pop(0)
                next_lines_num = next_lines_nums.pop(0)
                if self.simplify_line_chance > random_obj.random():
                    next_line = simplify_line(next_line, attr_dict, next_lines_num, random_obj)
                text_ids = text_ids + [self.sep_id] + self.tokenizer(next_line)['input_ids'][1:-1]
            else:
                break
        
        # trim to max_seq_length
        text_ids = text_ids[:self.max_seq_length]
        text_ids = torch.tensor(text_ids, dtype=torch.long)
        
        # labels
        labels = text_ids.clone().fill_(-100)
        labels[text_ids == self.mask_id] = character_id
        
        # segment ids [0 ... n], -1 = padding
        # each [sep] is a new segment
        segment_ids = torch.zeros_like(text_ids, dtype=torch.long).fill_(-100)
        prev_id_pos = 0
        prev_seg_id = 0
        for i, text_id in enumerate(text_ids):
            if text_id == self.sep_id:
                segment_ids[prev_id_pos:i].fill_(prev_seg_id)
                prev_id_pos = i
                prev_seg_id += 1
        
        # dropout text_ids
        if self.token_dropout > 0:
            text_ids[torch.randn_like(text_ids, dtype=torch.float) < self.token_dropout] = self.mask_id
        
        # attention mask
        attention_mask = torch.ones_like(text_ids)
        
        # pad to max_seq_length if needed
        if len(text_ids) < self.max_seq_length:
            pad = self.max_seq_length - len(text_ids)
            text_ids       = torch.nn.functional.pad(text_ids      , (0, pad), value=   0)
            segment_ids    = torch.nn.functional.pad(segment_ids   , (0, pad), value=-100)
            labels         = torch.nn.functional.pad(labels        , (0, pad), value=-100)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad), value=   0)
        
        return {
            'input_ids'     : text_ids,
            'segment_ids'   : segment_ids,
            'labels'        : labels,
            'attention_mask': attention_mask,
        }
    
    def __getitem__(self, idx):
        while True:
            try:
                return self._get_item(idx)
            except Exception as e:
                traceback.print_exc()
                idx += 1

    def __len__(self):
        return self.len


class LMDataModule(pl.LightningDataModule):
    def __init__(self, model_name_or_path, line_by_line, pad_to_max_length,
                 preprocessing_num_workers, overwrite_cache, max_seq_length,
                 train_batch_size, val_batch_size, dataloader_num_workers):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.line_by_line = line_by_line
        self.pad_to_max_length = pad_to_max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers

    def setup(self, stage=None):
        tokenizer = get_patched_distilbert()[1]
        
        if self.max_seq_length is None:
            self.max_seq_length = tokenizer.model_max_length
        else:
            if self.max_seq_length > tokenizer.model_max_length:
                warnings.warn(
                    f"The max_seq_length passed ({self.max_seq_length}) is larger than the maximum length for the"
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            self.max_seq_length = min(self.max_seq_length, tokenizer.model_max_length)
        
        self.train_dataset = CustomMLMDataset(tokenizer, self.max_seq_length, eval=False)
        self. eval_dataset = CustomMLMDataset(tokenizer, self.max_seq_length, eval=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.dataloader_num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.dataloader_num_workers,
            persistent_workers=True,
        )