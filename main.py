# The Plan:
# 1. Extract any speech data from fimfiction.net
# 2. Train a model on speech with "said X" or similar labels (with task of predicting X)
# 3. Add "said {maked}" to speech without any existing labels and use model to infer X
#    This can be done in stages, starting with the easiest labels and working up to the hardest
#    to increase the amount of context available to the model for each label

# adapted from https://github.com/yang-zhang/lightning-language-modeling/blob/main/language_model.py
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.nn.functional import mse_loss
from patch_model import get_patched_distilbert
from transformers.optimization import AdamW
from data import LMDataModule


class LMModel(pl.LightningModule):
    def __init__(self, learning_rate, adam_beta1, adam_beta2, adam_epsilon, regularization_loss):
        super().__init__()
        self.regularization_loss = regularization_loss
        self.pt_embed_weight = None
        self.save_hyperparameters() # note, all __init__ kwargs are loaded by self.save_hyperparameters(), somehow
        self.model, tokenizer = get_patched_distilbert()[:2]
    
    def forward(self, x):
        return self.model(x).logits
    
    def get_reg_loss(self):
        if self.pt_embed_weight is None:
            self.pt_embed_weight = get_patched_distilbert()[2]
            self.pt_embed_weight = self.pt_embed_weight.to(self.model.distilbert.embeddings.word_embeddings.weight.device)
        loss = mse_loss(self.model.distilbert.embeddings.word_embeddings.weight[:self.pt_embed_weight.shape[0]], self.pt_embed_weight)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        if self.regularization_loss:
            loss += self.get_reg_loss() * self.regularization_loss
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('valid_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          self.hparams.learning_rate,
                          betas=(self.hparams.adam_beta1,
                                 self.hparams.adam_beta2),
                          eps=self.hparams.adam_epsilon,)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=3.5e-5)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        return parser


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                        default="distilbert-base-cased")
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--line_by_line', action='store_true', default=False)
    parser.add_argument('--pad_to_max_length', action='store_true', default=False)
    parser.add_argument('--preprocessing_num_workers', type=int, default=4)
    parser.add_argument('--overwrite_cache', action='store_true', default=False)
    parser.add_argument('--max_seq_length'  , type=int, default=512)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size'  , type=int, default=16)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LMModel.add_model_specific_args(parser)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    data_module = LMDataModule(
        model_name_or_path=args.model_name_or_path,
        line_by_line=args.line_by_line,
        pad_to_max_length=args.pad_to_max_length,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    # ------------
    # model
    # ------------
    lmmodel = LMModel(
        args.learning_rate,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_epsilon,
        regularization_loss=0.01,
    )

    # ------------
    # checkpoint
    # ------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='valid_loss',
        dirpath='checkpoints',
        filename='lm-{epoch:02d}-{valid_loss:.2f}',
        save_top_k=10,
        mode='min',
        every_n_train_steps=2000,
        save_last=True,
    )
    
    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(lmmodel, data_module, ckpt_path=args.ckpt)


if __name__ == '__main__':
    cli_main()