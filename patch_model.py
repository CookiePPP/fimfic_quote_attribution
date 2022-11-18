# Using vocab.txt and vocab_pretrained.txt
# Modify model embeddings to include new characters
# (initializing the new vectors to semantically similar already existing vectors)
import torch

from utils import config
init_token = config['init_token']
vocab            = open('data/vocab.txt'           , 'r', encoding='utf8').read().splitlines()
vocab_pretrained = open('data/vocab_pretrained.txt', 'r', encoding='utf8').read().splitlines()
init_token_index = vocab.index(init_token)


def get_patched_distilbert(cased=True):
    from custom_model import CustomDistilBertForMaskedLM
    model = CustomDistilBertForMaskedLM.from_pretrained('distilbert-base-cased' if cased else 'distilbert-base-uncased')
    
    n_new_tokens = len(vocab) - len(vocab_pretrained)

    # add new tokens to embedding
    word_embed_data = model.distilbert.embeddings.word_embeddings.weight.data
    init_token = word_embed_data[init_token_index][None, :] # [n_vocab, n_dim] -> [1, n_dim]
    word_embed_data = torch.cat([word_embed_data, init_token.repeat(n_new_tokens, 1)], dim=0)
    model.distilbert.embeddings.word_embeddings.weight.data = word_embed_data
    
    # add new tokens to model output
    vocab_projector = model.vocab_projector
    if vocab_projector.weight.data.shape[0] == len(vocab_pretrained):
        vocab_projector.weight.data = torch.cat([vocab_projector.weight.data, torch.zeros(n_new_tokens, vocab_projector.weight.data.shape[1])], dim=0)
    vocab_projector.bias.data = torch.cat([vocab_projector.bias.data, torch.zeros(n_new_tokens)], dim=0)
    vocab_projector.out_features = len(vocab)
    
    # update config
    model.distilbert.embeddings.word_embeddings.num_embeddings = len(vocab)
    model.config.vocab_size = len(vocab)
    
    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer('data/vocab.txt', do_lower_case=not cased, mask_token='MASKTOKEN')
    
    from transformers import DistilBertForMaskedLM
    pt_embed_weight = DistilBertForMaskedLM.from_pretrained('distilbert-base-cased' if cased else 'distilbert-base-uncased')\
        .distilbert.embeddings.word_embeddings.weight.data
    
    return model, tokenizer, pt_embed_weight

if __name__ == '__main__':
    model, tokenizer = get_patched_distilbert()[:2]