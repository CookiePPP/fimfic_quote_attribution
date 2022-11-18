"""Get DistilBertForMaskedLM with segment embeddings added to the model"""
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertForMaskedLM
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.models.distilbert.modeling_distilbert import Embeddings
from typing import Optional, Union, Tuple

class CustomEmbeddings(Embeddings):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        n_segment_embeds = 7
        self.segment_embeddings = nn.Embedding(1 + n_segment_embeds, config.dim, padding_idx=0)
        self.n_segment_embeds = n_segment_embeds
    
    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.

        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        seq_length = input_ids.size(1)
        
        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)
        
        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        
        # Get segment embeddings
        segment_pad_mask = segment_ids < 0
        segment_ids = segment_ids.remainder(
            self.n_segment_embeds) + 1  # 0 is reserved for padding, above n embeds will wrap around
        segment_ids.masked_fill_(segment_pad_mask, 0)  # 0 is padding idx
        segment_embeddings = self.segment_embeddings(segment_ids)
        
        embeddings = word_embeddings + position_embeddings + segment_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class CustomDistilBertModel(DistilBertModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.embeddings = CustomEmbeddings(config)  # Embeddings
    
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            segment_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
        
        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids, segment_ids)  # (bs, seq_length, dim)
        return self.transformer(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CustomDistilBertForMaskedLM(DistilBertForMaskedLM):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.distilbert = CustomDistilBertModel(config)
    
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            segment_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        
        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        
        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return ((mlm_loss,) + output) if mlm_loss is not None else output
        
        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )