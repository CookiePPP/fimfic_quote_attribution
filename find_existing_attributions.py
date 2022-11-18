import unidecode
import glob
import json
import os
from utils import config, tqdm, strip_to_word, get_verbs_for_said

REBUILD_VOCAB = True


def split_first_clause(rest):
    eop_chars = {".", "?", "!", ";", ":", ","}
    first_clause = rest
    for i, c in enumerate(rest):
        if c in eop_chars:
            first_clause = rest[:i]
            break
        elif " and " == rest[i:i + 3]:
            first_clause = rest[:i]
            break
    return first_clause, rest[len(first_clause):]

def isname(name):
    """Can't use istitle() since it returns false for names like "Mane-iac"."""
    return name[0].isupper() and (len(name) == 1 or name[1:].islower())

def get_attr_from_line(line_str: str):
    # "Hi Applejack!" Rainbow Dash screamed. "How are you?"
    # "Hi Applejack!" screamed Rainbow Dash. "How are you?"
    # "Hi Applejack!" screamed Rainbow Dash.
    # "Hi Applejack!" screamed the mare. "How are you?"
    # screamed Rainbow Dash.
    # Hi Applejack" screamed Rainbow Dash.
    line_str = unidecode.unidecode(line_str).strip()
    
    # skip line_str with no speech
    if line_str.count('"') < 2:
        return None, []

    # -> [before quote 1, quote 1, after quote 1, quote 2, after quote 2]
    line_split = line_str.split('"')

    character = None
    verbs = []
    for after_quote in line_split[2::2]:
        verbs.append(None) # placeholder
        
        rest = after_quote.strip()
        
        # get substring of after_quote before any clause change
        # e.g: "Rainbow Dash asked, looking at her." -> "Rainbow Dash asked"
        rest = split_first_clause(rest)[0]

        words = rest.split()

        # if no speech verb in string, skip
        if not any(verb in words for verb in get_verbs_for_said()):
            continue
        
        # get character name (should be title format and next to verb)
        verb_pos = next(i for i, word in enumerate(words) if word in get_verbs_for_said())
        verbs[-1] = words[verb_pos]
        
        direction = 1
        if verb_pos > 0 and words[verb_pos-1].istitle():
            direction = -1
        
        character_split = []
        if direction == 1:
            for word in words[verb_pos + 1:]:
                if not isname(word):
                    break
                character_split.append(word)
        else:
            for word in words[verb_pos - 1::-1]:
                if not isname(word):
                    break
                character_split.insert(0, word)
        
        character_joined = " ".join(character_split).strip("\t \n\r'\"-")
        if character is None and character_joined and not character_joined.endswith("'s"):
            character = character_joined
    
    return character, verbs


# find all cases of "X said." or "X asked." or similar
# (with X being a character name and . being any sentence-ending punctuation)
# return line number, character name, and line text
def find_said_lines(lines):
    if type(lines) is str:
        lines = lines.splitlines()
    
    line_numbers = []
    characters_with_spaces = []
    characters = []
    line_texts = []
    for line_num, line_str in enumerate(lines):
        character, verbs = get_attr_from_line(line_str)
        
        # add to list
        if character:
            line_numbers.append(line_num)
            characters_with_spaces.append(character)
            characters.append(strip_to_word(character))
            line_texts.append(line_str)
    return line_numbers, characters_with_spaces, characters, line_texts

if __name__ == "__main__":
    fimfarchive_path = config['fimfarchive_path']
    
    txt_paths = glob.glob(os.path.join(fimfarchive_path, 'txt', '**', '*.txt'), recursive=True)
    attr_dict_path = config['attr_dict_path']
    
    attr_dict = {} # {path: {character: [line numbers]}}
    for txt_path in tqdm(txt_paths, smoothing=0.0):
        attr_dict[txt_path] = {}
        
        with open(txt_path, 'r', encoding='utf8') as f:
            txt_str = f.read()
        line_numbers, characters_with_spaces, characters, line_texts = find_said_lines(txt_str)
        
        for line_number, character_ws, line_text in zip(line_numbers, characters_with_spaces, line_texts):
            if character_ws in config['character_ignore_list_train']:
                continue
            if character_ws not in attr_dict[txt_path]:
                attr_dict[txt_path][character_ws] = []
            attr_dict[txt_path][character_ws].append(line_number)
        
        # replace character names with spaces with underscores
        #for character_with_spaces, character in zip(characters_with_spaces, characters):
        #    txt_str = txt_str.replace(character_with_spaces, character)
    
    min_lines = config['min_character_samples']
    character_n_lines = {} # {character: number of lines}
    for txt_path in attr_dict:
        for character_ws in attr_dict[txt_path]:
            if character_ws not in character_n_lines:
                character_n_lines[character_ws] = 0
            character_n_lines[character_ws] += len(attr_dict[txt_path][character_ws])
    attr_dict = {txt_path: {character_ws: attr_dict[txt_path][character_ws] for character_ws in attr_dict[txt_path] if character_n_lines[character_ws] >= min_lines} for txt_path in attr_dict}
    
    json.dump(attr_dict, open(attr_dict_path.replace('.pt', '.json'), 'w', encoding='utf-8'), indent=4)
    
    character_ws_list = sorted(list(character_n_lines.keys()), key=lambda x: character_n_lines[x], reverse=True)
    character_ws_list = [character_ws for character_ws in character_ws_list if character_n_lines[character_ws] >= min_lines]
    character_list = [strip_to_word(x) for x in character_ws_list]
    character_n_lines_list = [character_n_lines[x] for x in character_ws_list]
    json.dump(character_ws_list, open('character_list.json', 'w', encoding='utf-8'), indent=4)
    
    if REBUILD_VOCAB:
        __import__("rebuild_vocab") # run rebuild_vocab.py