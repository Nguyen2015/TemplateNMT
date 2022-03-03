
from typing import Dict, List, Optional 
from fairseq.models.transformer import DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS, TransformerEncoder, TransformerModel
import random
from fairseq.modules.multihead_attention import MultiheadAttention


import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


from fairseq.modules import (
    TransformerEncoderLayer, LayerNorm, transformer_layer,
)

from fairseq.models import (
    register_model,
    register_model_architecture,
)


from fairseq.models.transformer import (
    TransformerDecoder,
    base_architecture, transformer_iwslt_de_en, transformer_wmt_en_de,
    transformer_vaswani_wmt_en_de_big, transformer_vaswani_wmt_en_fr_big, transformer_wmt_en_de_big, transformer_wmt_en_de_big_t2t
)

 

@register_model('multisources_transformer')
class MultisourcesTransformerModel(TransformerModel): 
 
    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--template-dropout', type=float,
                            default=0.5,
                            help='template dropout rate')    
        parser.add_argument('--template-src-attn',
                            default=False, action='store_true',
                            help='learn (template - source) attention')    
        parser.add_argument('--no-gating-src-templ',
                            default=False, action='store_true',
                            help='not use gating when incorporting source and templ')    

    @classmethod
    def build_model(cls, args, task):

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict, template_dictionary = task.source_dictionary, task.target_dictionary, task.template_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        template_embed_tokens = cls.build_embedding(args,
            template_dictionary, args.encoder_embed_dim,
        )

        # MultisourcesTransformerEncoder
        encoder = MultisourcesTransformerEncoder(
            args, src_dict, encoder_embed_tokens, template_dictionary, template_embed_tokens)
        decoder = MultisourcesTransformerDecoder(
            args, tgt_dict, decoder_embed_tokens)
        # 
 
        return cls(args, encoder, decoder)
 
    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        template_tokens,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, template_tokens=template_tokens, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

class MultisourcesTransformerEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``True``
    """

    def __init__(self, args, dictionary, embed_tokens, template_dictionary, template_embed_tokens, left_pad=True):
        super().__init__(args, dictionary, embed_tokens)

        self.dropout = args.dropout

        # for template
        self.template_dropout = args.template_dropout # template dropout mechanism
        self.template_dictionary = template_dictionary 
        self.embed_template = template_embed_tokens
        self.template_layers = nn.ModuleList([])
        self.template_layers.extend(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)])

        self.learn_template_src_attn = args.template_src_attn
        if args.template_src_attn:
            self.template_src_attn = MultiheadAttention(
                embed_tokens.embedding_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
                q_noise=getattr(args, "quant_noise_pq", 0), 
                qn_block_size=getattr(args, "quant_noise_pq_block_size", 8)
            )

        self.normalize = args.encoder_normalize_before
        if args.encoder_normalize_before:
            self.template_layer_norm = LayerNorm(embed_tokens.embedding_dim)

    def forward(self, src_tokens, src_lengths, template_tokens, return_all_hiddens=False):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            template_tokens (LongTensor): template tokens of shape
                `(batch, templ_len)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """

        def _enc_source(sequence_input, enc_layers, layer_norm, embed_layer, return_all_hiddens=False):
            # embed tokens and positions and tags
            x = self.embed_scale * embed_layer(sequence_input)
            if self.embed_positions is not None:
                x += self.embed_positions(sequence_input)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

            encoder_states = []

            if return_all_hiddens:
                encoder_states.append(x)

            # compute padding mask
            encoder_padding_mask = sequence_input.eq(self.padding_idx)
            if not encoder_padding_mask.any():
                encoder_padding_mask = None

            # encoder layers
            for enc_layer in enc_layers:
                x = enc_layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

            if self.normalize:
                x = layer_norm(x)

            return x, encoder_padding_mask, encoder_states

        sentence_out, sentence_padding_mask, encoder_states = _enc_source(
            src_tokens, self.layers, self.layer_norm if self.normalize else None, self.embed_tokens, 
            return_all_hiddens=return_all_hiddens)
        template_out, template_padding_mask = None, None
        
        random_variable = random.randrange(0, 100, 1) / 100
        template_states = []
        if random_variable > self.template_dropout or not self.training:
            template_out, template_padding_mask, template_states = _enc_source(
                template_tokens, self.template_layers, self.template_layer_norm if self.normalize else None, self.embed_template, 
                return_all_hiddens=return_all_hiddens)
                
            if self.learn_template_src_attn:
                sentence_out, template_src_attn = self.template_src_attn(
                        query=sentence_out,
                        key=template_out,
                        value=template_out,
                        key_padding_mask=template_padding_mask,
                        incremental_state=None,
                        static_kv=True,
                        need_weights=False,
                        need_head_weights=False,
                    )

        return {
            'encoder_out': [sentence_out],  # T x B x C
            'encoder_padding_mask': [sentence_padding_mask],  # B x T
            'template_out': [template_out],  # T x B x C
            'template_padding_mask': [template_padding_mask],  # B x T
            "encoder_states": encoder_states,  # List[T x B x C]
            "template_states": template_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        # result = super().reorder_encoder_out(encoder_out, new_order)
        result = {}
        if len(encoder_out["encoder_out"]) == 0:
            encoder_out_new = []
        else:
            encoder_out_new = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            encoder_padding_mask = []
        else:
            if encoder_out["encoder_padding_mask"][0] is not None:
                encoder_padding_mask = [
                                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
                            ] 
            else:
                encoder_padding_mask = [None]
        if len(encoder_out["template_out"]) == 0:
            template_out = []
        else:
            if encoder_out["template_out"][0] is not None:
                template_out = [encoder_out["template_out"][0].index_select(1, new_order)]
            else:
                template_out = [None] 


        if len(encoder_out["template_padding_mask"]) == 0:
            template_padding_mask = []
        else:
            if encoder_out["template_padding_mask"][0] is not None:
                template_padding_mask = [
                    encoder_out["template_padding_mask"][0].index_select(0, new_order)
                ] 
            else: 
                template_padding_mask = [None] 

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]
                
        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        template_states = encoder_out["template_states"]
        if len(template_states) > 0:
            for idx, state in enumerate(template_states):
                template_states[idx] = state.index_select(1, new_order)
        
            
        result['encoder_out'] = encoder_out_new
        result['template_out'] = template_out
        result['encoder_padding_mask'] = encoder_padding_mask
        result['template_padding_mask'] = template_padding_mask
        result['encoder_states'] = encoder_states
        result['template_states'] = template_states
        result['src_tokens'] = src_tokens
        result['src_lengths'] = src_lengths
        return result

class MultisourcesTransformerDecoderLayer(transformer_layer.TransformerDecoderLayer):
    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__( 
            args,
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args
        self.template_encoder_attn = self.build_encoder_attention(self.embed_dim, args)
        
        if not self.args.no_gating_src_templ:
            self.weight_template = nn.Linear(self.embed_dim, self.embed_dim)
            self.weight_source = nn.Linear(self.embed_dim, self.embed_dim)
            self.rating_calculator = nn.Sigmoid()
 
    
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        enc_template=None,
        template_padding_mask=None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False if self.training else True,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )

            # template encoding 
            if enc_template is not None:
                x_template, attn_template = self.template_encoder_attn(
                    query=x,
                    key=enc_template,
                    value=enc_template,
                    key_padding_mask=template_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=need_attn or (not self.training and self.need_attn),
                    need_head_weights=need_head_weights,
                )
                if self.args.no_gating_src_templ:
                    x = 0.5*(x + x_template)
                else: 
                    source_rate = self.rating_calculator(self.weight_template(x) + self.weight_source(x_template))
                    x = source_rate*x + (1 - source_rate)*x_template
            ##

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

class MultisourcesTransformerDecoder(TransformerDecoder): 

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return MultisourcesTransformerDecoderLayer(args, no_encoder_attn)
        
    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"

            # get template encoding output vector
            enc_template = encoder_out.get("template_out", [None])[0]
            template_padding_mask = encoder_out.get("template_padding_mask", [None])[0]
            if enc_template is not None:
                enc_template = encoder_out["template_out"][0]
                assert (
                    enc_template.size()[1] == bs
                ), f"Expected enc_template.shape == (t, {bs}, c) got {enc_template.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                enc_template=enc_template,  # encoding information of template
                template_padding_mask=template_padding_mask # encoding information of template 
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

@register_model_architecture('multisources_transformer', 'multisources_transformer')
def pascal_base_architecture(args):
    base_architecture(args)


@register_model_architecture('multisources_transformer', 'multisources_transformer_iwslt_de_en')
def pascal_transformer_iwslt_de_en(args):
    transformer_iwslt_de_en(args)


@register_model_architecture('multisources_transformer', 'multisources_transformer_wmt_en_de')
def multisources_transformer_wmt_en_de(args):
    transformer_wmt_en_de(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani, et al, 2017)
@register_model_architecture('multisources_transformer', 'multisources_transformer_vaswani_wmt_en_de_big')
def multisources_transformer_vaswani_wmt_en_de_big(args):
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('multisources_transformer', 'multisources_transformer_vaswani_wmt_en_fr_big')
def multisources_transformer_vaswani_wmt_en_fr_big(args):
    transformer_vaswani_wmt_en_fr_big(args)


@register_model_architecture('multisources_transformer', 'multisources_transformer_wmt_en_de_big')
def multisources_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('multisources_transformer', 'multisources_transformer_wmt_en_de_big_t2t')
def multisources_transformer_wmt_en_de_big_t2t(args):
    transformer_wmt_en_de_big_t2t(args)
