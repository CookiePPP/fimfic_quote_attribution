import os.path
import json
import glob
from infer import load_model, load_text_for_model, infer
from utils import tqdm

infer_kwargs = dict(
    search_length=4,
    batch_size=64,
    fast_fill_threshold=0.95,
    max_seq_length=512,
    mask_mode=31,
    use_existing_verbs=False,
    default_verb='said',
    use_pbar=False,
)

model_paths = [
    "checkpoints/run_04_256context/lm-epoch=01-valid_loss=0.77.ckpt",
]
model_paths.extend(glob.glob("checkpoints/*.ckpt"))
device = 'cuda'
dtype = 'float'
file_paths = [
    'test_model/ahodc.txt',
    'test_model/ss_ch3.txt',
]

best_infer_config = None
best_accuracy = 0
best_model_path = None
config_scores = {}
assert all(os.path.exists(p) for p in model_paths), f"Missing model paths: {[p for p in model_paths if not os.path.exists(p)]}"
for model_path in tqdm(model_paths, desc="Model", smoothing=0.1):
    if not os.path.exists(model_path):
        continue
    model, tokenizer = load_model(model_path, device, dtype)
    
    for mask_mode in [20, 21, 30, 31]:
        infer_kwargs['mask_mode'] = mask_mode
        for default_verb in ['said']:
            infer_kwargs['default_verb'] = default_verb
            
            if 'context/' in model_path:
                max_seq_length = int(model_path.split('context/')[0].split('_')[-1])
                infer_kwargs['max_seq_length'] = max_seq_length
            else:
                infer_kwargs['max_seq_length'] = 512
            
            n_correct = 0
            n_checked = 0
            for file_path in file_paths:
                file_path_without_ext = os.path.splitext(file_path)[0]
                text = load_text_for_model(file_path)
                lines = [line.strip() for line in text.splitlines()]
                attrs, mean_confidence = infer(lines, tokenizer, model, **infer_kwargs)
                #json.dump(attrs, open(f'{file_path_without_ext}_pred.json', 'w'), indent=2)
                
                answer_path = f'{file_path_without_ext}_answers.json'
                answers = json.load(open(answer_path, 'r', encoding='utf8'))
                for (correct_character, _, line_num) in answers:
                    assert attrs[line_num][2] == line_num
                    pred_character = attrs[line_num][0]
                    n_checked += 1
                    if pred_character == correct_character:
                        n_correct += 1
                    if pred_character != correct_character and 0:
                        print(f'{line_num:03} Fail: {pred_character:>10} != {correct_character}')
            accuracy = n_correct / n_checked
            
            tqdm.write(f"\"{infer_kwargs['mask_mode']}\" {default_verb:7} " + f'{n_correct:>3} / {n_checked:>3} correct ({accuracy:.1%})')
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_infer_config = infer_kwargs.copy()
                best_model_path = model_path

print(f"Best config: {best_infer_config}\nBest Model: {best_model_path}\nwith accuracy {best_accuracy:.1%}")