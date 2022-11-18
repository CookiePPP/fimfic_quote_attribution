import argparse
import os
import json
import shutil

from utils import config, tqdm
from unidecode import unidecode

def extract_epub(epub_path):
    """Extracts epub file to directory with same name as epub file"""
    import zipfile
    with zipfile.ZipFile(epub_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.splitext(epub_path)[0])

def get_chapter_filenames(opf_path):
    import xml.etree.ElementTree as ET
    if not os.path.exists(opf_path):
        opf_path = os.path.join(os.path.dirname(opf_path), "book.opf")
    tree = ET.parse(opf_path)
    root = tree.getroot()
    manifest = root.find('{http://www.idpf.org/2007/opf}manifest')
    hrefs = [item.attrib['href'] for item in manifest.findall('{http://www.idpf.org/2007/opf}item')]
    hrefs = [href for href in hrefs if href.endswith('.html')]
    return hrefs

def get_text_from_html(html_path):
    """Returns text from html file with paragraphs separated by newlines"""
    from bs4 import BeautifulSoup
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    texts = []
    for p in soup.find_all('p'):
        texts.extend([BeautifulSoup(l, 'html.parser').get_text() for l in str(p).split('<br/>')])
    text = "\n".join(texts)
    
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    return text

def get_text_from_epub(epub_path):
    extract_epub(epub_path)
    opf_path = os.path.join(os.path.splitext(epub_path)[0], "content.opf")
    chapter_filenames = get_chapter_filenames(opf_path)
    assert len(chapter_filenames) > 0, f"No chapters found in epub\n{epub_path}"
    texts = []
    for chapter_filename in chapter_filenames:
        html_path = os.path.join(os.path.splitext(epub_path)[0], chapter_filename)
        texts.append(get_text_from_html(html_path))
    text = "\n".join(texts)
    text = unidecode(text)
    while "  " in text:
        text = text.replace("  ", " ")
    
    #shutil.rmtree(os.path.splitext(epub_path)[0])  # delete extracted epub
    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--n_ranks', type=int, default=1)
    args = parser.parse_args()
    
    # This file will take fimfarchive epub files and convert them to txt files
    
    fimfarchive_path = config['fimfarchive_path']
    index_path = os.path.join(fimfarchive_path, "index.json")
    epubs_dir = os.path.join(fimfarchive_path, "epub")
    txts_dir = os.path.join(fimfarchive_path, "txt") # this is output dir for this script
    os.makedirs(txts_dir, exist_ok=True)
    
    index = json.load(open(index_path, "r", encoding='utf8'))
    story_ids = list(index.keys())[args.rank::args.n_ranks]
    for story_id in tqdm(story_ids, smoothing=0.0):
        story = index[story_id]
        epub_path = os.path.join(fimfarchive_path, story["archive"]["path"])
    
        text = get_text_from_epub(epub_path)
    
        txt_dir = os.path.join(txts_dir, story_id[:2])
        txt_path = os.path.join(txt_dir, story_id + ".txt")
        if os.path.exists(txt_path):
            continue
    
        os.makedirs(txt_dir, exist_ok=True)
        with open(txt_path, "w", encoding='utf-8') as f:
            f.write(text)
    
    # if more than one speech on a line, split into separate lines
    # Luna looked at you. "Hi" said Luna. "How are you?"
    # [becomes]
    # Luna looked at you.
    # "Hi" said Luna.
    # "How are you?"
    #for i, story_id in enumerate(tqdm(index, smoothing=0.01)):
    #    if i < 18882:
    #        continue
    #    story = index[story_id]
    #    txt_path = os.path.join(txts_dir, story_id[:2], story_id + ".txt")
    #    with open(txt_path, "r", encoding='utf-8') as f:
    #        text = f.read()
    #    new_lines = []
    #    for line in text.splitlines():
    #        if '"' not in line:
    #            new_lines.append(line)
    #            continue
    #        # 1. "Hi" said Luna. "How are you?"
    #        # 2. ["", "Hi", " said Luna. ", "How are you?", ""]
    #        line_split = line.split('"')
    #        quotes = [line_split[0], ]
    #        for i in range(1, len(line_split), 2):
    #            quotes.append("\""+"\"".join(line_split[i:i+2]))
    #        quotes = [s for s in quotes if s.strip()]
    #        new_lines.extend(quotes)
    #    new_lines = "\n".join(new_lines)
    #    try:
    #        with open(txt_path, "w", encoding='utf-8') as f:
    #            f.write(new_lines)
    #    except KeyboardInterrupt:
    #        with open(txt_path, "w", encoding='utf-8') as f:
    #            f.write(new_lines)
    #        raise KeyboardInterrupt