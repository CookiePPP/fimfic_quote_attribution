import os
import glob
import re
import textwrap

import torch
from unidecode import unidecode

from create_dataset import get_text_from_epub, get_text_from_html
from find_existing_attributions import get_attr_from_line, split_first_clause
from patch_model import get_patched_distilbert
from utils import tqdm, get_implications_ids, get_female_character_ids, get_character_blacklist_ids, zip_equal, config, get_implications


# It was a sunny day.
# "Hi Applejack!" screamed Rainbow Dash. "How are you?"
# "I'm fine, Rainbow Dash"

# narrator: It was a sunny day.
# [NEW_LINE]
# rainbow dash: (screamed) Hi Applejack!
# narrator: screamed Rainbow Dash.
# rainbow dash: How are you?
# [NEW_LINE]
# applejack: I'm fine, Rainbow Dash

def load_text_for_model(path):
    """Load text from a txt or epub file and pre-process."""
    try:
        if path.endswith('.txt'):
            text = open(path, 'r', encoding='utf8').read()
        elif path.endswith('.epub'):
            text = get_text_from_epub(path)
        elif path.endswith('.html'):
            text = get_text_from_html(path)
        else:
            raise ValueError("Invalid file type: " + path)
    except Exception as e:
        print(f'Error loading {path}: {e}')
        raise e
    
    # remove/normalize special characters
    text = unidecode(text)
    
    # remove extra whitespace
    text = re.sub(' {2,}', ' ', text)
    
    # remove blank lines
    text = "\n".join([t for t in text.splitlines() if t.strip()])
    
    return text

def get_model_input(tokenizer, lines, verb, line_num, max_seq_length=512, mask_mode=1, use_existing_verbs=False, default_verb='said'):
    """
    Returns text_ids, attention_mask, line which can be batched and fed into model.
    """
    sep_id = tokenizer.sep_token_id
    mask_id = tokenizer.mask_token_id
    mask_token = tokenizer.mask_token
    
    # get line (which will be modified with a [MASK])
    line = lines[line_num]
    # e.g: '"Hi Applejack!" she said.'
    # e.g: "Rarity," She replied flatly, "You have a store in Canterlot."
    # e.g: "Yes"
    # In some cases, 'she' or 'She' needs to be replaced with '[MASK]'.
    # In other cases, adding a new clause is necessary.
    
    # TODO: check if this func improves results or not.
    line_masked = None
    if use_existing_verbs and verb is not None:
        # if existing speech verb, replace with [MASK]
        # e.g: '"Yes" she said.' -> '"Yes" [MASK] said.'
        # e.g: '"Yes" She replied flatly.' -> '"Yes" [MASK] replied flatly.'
        
        line_split = line.split('"')
        # -> [before quote 1, quote 1, after quote 1, quote 2, after quote 2, ...]
        for i, part in enumerate(line_split):
            if i % 2 == 1 or i == 0:
                continue
            if verb not in part:
                continue
            # replace substring before clause change with f'verb [MASK].'
            part1, part2 = split_first_clause(part)
            part1 = f' {mask_token} {verb}.'
            line_split[i] = part1 + part2
            break
        line_masked = '"'.join(line_split)
    
    if line_masked is None:
        # if no existing speech verb, add a new clause or replace old one
        # e.g: '"Yes"' -> '"Yes" said [MASK].'
        
        line_split = line.split('"')
        # -> [before quote 1, quote 1, after quote 1, quote 2, after quote 2, ...]
        
        # e.g: '"Yes" Applejack blinked. "I'm sure"' -> '"Yes" said [MASK]. Applejack blinked. "I'm sure"'
        if   mask_mode == 10:
            line_split[2] = f' {mask_token} {default_verb}.' + line_split[2]
            line_masked = '"'.join(line_split)
        
        # e.g: '"Yes" Applejack blinked. "I'm sure"' -> '"Yes" said [MASK]. Applejack blinked. "I'm sure"'
        elif mask_mode == 11:
            line_split[-1] = f' {mask_token} {default_verb}.' + line_split[2]
            line_masked = '"'.join(line_split)
        
        # e.g: '"Yes" Applejack blinked. "I'm sure"' -> '"Yes" said [MASK]. "I'm sure"'
        elif mask_mode == 20:
            line_split[2] = f' {mask_token} {default_verb}. '
            line_masked = '"'.join(line_split).strip()
        
        # e.g: '"Yes" Applejack blinked. "I'm sure"' -> '"Yes" said [MASK]. "I'm sure"'
        elif mask_mode == 21:
            line_split[-1] = f' {mask_token} {default_verb}. '
            line_masked = '"'.join(line_split).strip()
        
        # e.g: '"Yes" Applejack blinked. "I'm sure"' -> '"Yes" said [MASK].'
        elif mask_mode == 30:
            line_masked = f'"{line_split[1]}" {mask_token} {default_verb}.'
        
        # e.g: '"Yes" Applejack blinked. "I'm sure"' -> '"Yes" said [MASK].'
        elif mask_mode == 31:
            line_masked = f'"{line_split[-2]}" {mask_token} {default_verb}.'
        
        # e.g: '"Yes" Applejack blinked. "I'm sure"' -> '"Yes" said [MASK].'
        elif mask_mode == 40:
            line_masked = f'{line_masked} {mask_token} {default_verb}.'
        else:
            raise ValueError(f'Invalid mask_mode: {mask_mode}')
    
    assert mask_token in line_masked, f'"{mask_token}" (mask token) not found in line_masked below:\n{line_masked}'
    assert mask_token in line_masked, f'"{mask_token}" (mask token) not found in line_masked below:\n{line_masked}'
    assert line_masked is not None, f'line_masked is None'
    
    text_ids = tokenizer(line_masked)['input_ids'][1:-1]  # list[int]
    assert mask_id in text_ids, f'mask_id ({mask_id}) not found in text_ids\n{text_ids}\n{line_masked}'
    
    # randomly add a line from the previous lines or the next lines till target length is reached
    prev_lines = lines[:line_num]
    next_lines = lines[line_num + 1:]
    prev_line_len = 0
    next_line_len = 0
    while len(text_ids) < max_seq_length:
        if prev_lines and (prev_line_len <= next_line_len or not next_lines):
            prev_line = prev_lines.pop(-1)
            text_ids = tokenizer(prev_line)['input_ids'][1:-1] + [sep_id] + text_ids
            prev_line_len += len(prev_line)
        elif next_lines:
            next_line = next_lines.pop(0)
            text_ids = text_ids + [sep_id] + tokenizer(next_line)['input_ids'][1:-1]
            next_line_len += len(next_line)
        else:
            break
    
    # trim to max_seq_length
    if len(text_ids) > max_seq_length:
        text_ids = text_ids[:max_seq_length] if next_line_len > prev_line_len else text_ids[-max_seq_length:]
    assert mask_id in text_ids, f'mask_id ({mask_id}) not found in text_ids after trim operation.'
    
    # convert to tensor
    text_ids = torch.tensor(text_ids, dtype=torch.long)  # [max_seq_length]
    
    # attention mask
    attention_mask = torch.ones_like(text_ids)  # [max_seq_length]
    
    # pad to max_seq_length if needed
    if len(text_ids) < max_seq_length:
        pad = max_seq_length - len(text_ids)
        text_ids = torch.nn.functional.pad(text_ids, (0, pad), value=0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad), value=0)
    
    return {
        'text_ids': text_ids,
        'attention_mask': attention_mask,
        'line_masked': line_masked,
        'line_num': line_num,
    }

def collate_fn(batch):
    """Collate function for batching."""
    return {
        'text_ids': torch.stack([b['text_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'lines_masked': [b['line_masked'] for b in batch],
        'line_nums': [b['line_num'] for b in batch],
    }

def apply_implications(probs: torch.Tensor):
    """
    Take a [B, n_vocab] tensor of probabilities and merge probabilities of specied tokens.
    """
    for (src_id, tgt_id) in get_implications_ids():
        probs[:, tgt_id] += probs[:, src_id]
        probs[:, src_id] = 0

def apply_blacklist(logits: torch.Tensor):
    """
    Take a [B, n_vocab] tensor of logit scores and set the logit scores of specied tokens to -inf.
    """
    for id in get_character_blacklist_ids():
        logits[:, id] = -float('inf')

def apply_female_blacklist(logits: torch.Tensor):
    for id in get_female_character_ids():
        logits[:, id] = -float('inf')

def infer_speaker_of_line(tokenizer, model, text_ids, attention_mask, batch_lines_masked):
    """
    Takes list of lines and line number and returns speaker.
    Uses neural network to infer speaker when ambiguous.
    """
    mask_id = tokenizer.mask_token_id
    mask_token = tokenizer.mask_token
    
    # run model
    with torch.no_grad():
        device = next(model.parameters()).device
        logits = model(
            text_ids      .to(device=device),
            attention_mask.to(device=device)
            )['logits'].float()
    
    # extract logits for [MASK] token
    logits = logits[text_ids == mask_id, :] # [B, n_masks, vocab_size] -> [B, vocab_size]
    logits = logits.cpu()
    B, n_vocab = logits.shape
    assert B == len(batch_lines_masked), f'Mismatch! Got {B} masked tokens and {len(batch_lines_masked)} batch size.'
    
    # get top prediction
    apply_blacklist(logits)
    probs = torch.softmax(logits, dim=1) # [B, vocab_size]
    apply_implications(probs)
    confidences, indexes = probs.max(dim=1) # -> [B], [B]
    
    # convert to speaker name
    speakers = []
    batch_lines_predicted = []
    for i, (line, index, confidence) in enumerate(zip(batch_lines_masked, indexes.unbind(0), confidences.unbind(0))):
        speaker = tokenizer._convert_id_to_token(index.item())
        batch_lines_predicted.append(line.replace(mask_token, speaker))
        speakers.append(speaker)
    
    return speakers, confidences, batch_lines_predicted

def infer(lines, tokenizer, model, search_length=8, batch_size=64, fast_fill_threshold=0.99, max_seq_length=256, mask_mode=1, use_existing_verbs=False, default_verb='said', use_pbar=True):
    """
    Takes text and returns ordered list of (speaker, speech_verb, line_num) tuples.
    Uses neural network to infer speaker or speech_verb when ambiguous.
    """
    lines = lines.copy()
    
    line_attrs = [] # [[speaker, speech_verb, line_num], ...]
    
    # split into (speaker, speech_verb, line) tuples
    for line_num, line in enumerate(lines):
        character, verbs = get_attr_from_line(line) # get speaker and speech_verb (if speaker is obvious)
        if character in config['character_blacklist_infer']:
            character = None
        verb = ([v for v in verbs if v]+[None])[0] # grab first verb (if any)
        line_attrs.append([character, verb, line_num])
    
    # assign "Narrator" or "UnknownSpeaker" to None speakers
    for i, (character, verb, line_num) in enumerate(line_attrs):
        if character is None:
            has_speech = lines[line_num].count('"') > 1
            line_attrs[i][0] = "UnknownSpeaker" if has_speech else "Narrator"
    
    # identify speaker for lines with "UnknownSpeaker"
    # (use neural network to infer speaker. start with easiest lines and repeat until all lines are assigned)
    missing_lines = [t for t in line_attrs if t[0] == "UnknownSpeaker"]
    if use_pbar:
        pbar = tqdm(total=len(missing_lines), desc='Inferring speakers', leave=False)
    confidences = []
    for _ in range(len(missing_lines)):
        if len(missing_lines) == 0:
            break
        
        # infer every line with a missing speaker and find the highest confidence prediction
        results = []
        batch = []
        for i, (character, verb, line_num) in enumerate(missing_lines[:search_length]):
            is_last_item = i == len(missing_lines[:search_length])-1
            item = get_model_input(tokenizer, lines, verb, line_num,
               max_seq_length=max_seq_length, mask_mode=mask_mode,
               use_existing_verbs=use_existing_verbs, default_verb=default_verb)
            batch.append(item)
            if len(batch) == batch_size or is_last_item:
                batch = collate_fn(batch)
                speaker_batch, confidence_batch, lines_pred_batch = infer_speaker_of_line(
                    tokenizer, model, batch['text_ids'], batch['attention_mask'], batch['lines_masked'])
                for result in zip_equal(speaker_batch, confidence_batch, batch['line_nums'], lines_pred_batch):
                    results.append(result)
                batch = []
        
        # update with results
        results = sorted(results, key=lambda x: x[1], reverse=True) # sort highest confidence first
        new_line_nums = []
        for i, (speaker, conf, line_num, line_pred) in enumerate(results):
            is_first_item = bool(i == 0)
            if not is_first_item and conf < fast_fill_threshold:
                break
            #tqdm.write(f"Labelled line {line_num:>4} with {speaker} (confidence: {conf:2.0%})\n{lines[line_num]}\n{line_pred}\n")
            if use_pbar: pbar.update(1)
            line_attrs[line_num][0] = speaker
            lines[line_num] = line_pred
            new_line_nums.append(line_num)
            confidences.append(conf)
        # remove labelled lines from missing_lines
        missing_lines = list(filter(lambda x: x[2] not in new_line_nums, missing_lines))
    if use_pbar: pbar.close()
    
    implications = {k: v for k, v in get_implications()}
    for i, (character, verb, line_num) in enumerate(line_attrs):
        line_attrs[i][0] = implications.get(character, character)
    
    mean_confidence = sum(confidences) / len(confidences)
    
    return line_attrs, mean_confidence


def get_script_from_line_attrs(line_attrs, lines):
    """
    Takes line_attrs and lines and returns play-script formatted string.
    """
    out_lines = []
    for character, verb, line_num in line_attrs:
        line = lines[line_num].strip()
        line_split = line.split('"')
        # -> [before quote 1, quote 1, after quote 1, quote 2, after quote 2, ...]
        for i, part in enumerate(line_split):
            part = part.strip()
            if not part:
                continue
            if i % 2 == 0:
                out_lines.append(f'{"Narrator"} ({None}) : {part}')
            else:
                out_lines.append(f'{character} ({verb}) : {part}')
        out_lines.append('')
    return '\n'.join(out_lines)

def load_model(model_path, device, dtype):
    model, tokenizer = get_patched_distilbert()[:2]
    model.load_state_dict({k[6:]: v for k, v in torch.load(model_path)['state_dict'].items()})
    model.eval().to(device).to(dtype=getattr(torch, dtype))
    return model, tokenizer

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    bool_fn = lambda x: x.lower() in ['true', '1', 't', 'y', 'yes']
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--from_file' , type=str, default=None)
    parser.add_argument('--from_dir'  , type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--skip_existing', type=bool_fn, default=True)
    parser.add_argument('--label_with_conf', type=bool_fn, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='half')
    args = parser.parse_args()

    infer_kwargs = dict(
        search_length=32,
        batch_size=64,
        fast_fill_threshold=0.99,
        max_seq_length=256,
        mask_mode=21,
        use_existing_verbs=False,
        default_verb='said',
    )
    
    model, tokenizer = load_model(args.model_path, args.device, args.dtype)
    
    files = []
    if args.from_file is not None:
        files.append(args.from_file)
    if args.from_dir is not None:
        files.extend(glob.glob(os.path.join(args.from_dir, '*.txt')))
        files.extend(glob.glob(os.path.join(args.from_dir, '*.epub')))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    file_sizes = [os.path.getsize(f)//1024 for f in files]
    pbar = tqdm(total=sum(file_sizes), desc='Processing files', unit='KB', smoothing=0.0)
    confidences = []
    for i, file in enumerate(files):
        filename = os.path.basename(file)
        filename_no_ext = os.path.splitext(filename)[0]
        output_file = os.path.join(args.output_dir, filename_no_ext + '.txt')
        if args.skip_existing and os.path.exists(output_file):
            tqdm.write(f"Skipping {filename} (already exists)")
            pbar.update(file_sizes[i])
            continue
        
        try:
            text = load_text_for_model(file)
            lines = [line.strip() for line in text.splitlines()]
            attrs, mean_confidence = infer(lines, tokenizer, model, **infer_kwargs)
            script_text = get_script_from_line_attrs(attrs, lines)
        except Exception as e:
            tqdm.write(f"Error processing {filename}: {e}")
            raise e
        open(output_file, 'w').write(script_text)
        tqdm.write(f"Saved {filename} with mean confidence {mean_confidence:2.0%}")
        confidences.append(mean_confidence)
        pbar.update(file_sizes[i])
    
    if len(confidences):
        tqdm.write(f"Mean confidence of all files: {sum(confidences) / len(confidences):2.1%}")