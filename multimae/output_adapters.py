# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv MAE, DPT and ConvNeXt code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# https://github.com/isl-org/DPT
# https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .multimae_utils import (Block, CrossAttention, Mlp, Attention, DecoderBlock,
                             build_2d_sincos_posemb, pair, trunc_normal_)
from .output_adapter_utils import (ConvNeXtBlock, Interpolate,
                                   make_fusion_block, make_scratch)


class SpatialOutputAdapter(nn.Module):
    """Cross-attention adapter for spatial outputs, like images or feature maps.

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens_enc: Dimension of tokens coming from encoder. Can be set using init method.
    :param dim_tokens_enc: Dimension of decoder tokens
    :param depth: Number of additional (full self-attention) transformer layers after initial cross attention and MLP
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    :param mlp_ratio: MLP hidden dim ratio
    :param num_heads: Number of attention heads
    :param qkv_bias: Set to True to enable bias
    :param drop_rate: Probability of dropping attention layer outputs
    :param attn_drop_rate: Probability of dropping attention matrix elements
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    :param use_task_queries: When set to True, adds task specific tokens from encoder (if available)
        to the corresponding query entries
    :param task: Task for which encoder tokens are added to the queries of the decoder (e.g. RGB if decoder is used for RGB)
    :param context_tasks: Tasks / modalities from the encoder. Used to create learned embeddings for each task.
    :param use_xattn: When set to True, attend to the tokens from the encoder through a cross-attention layer
    """

    def __init__(self,
                 num_channels: int,
                 stride_level: int,
                 patch_size_full: Union[int, Tuple[int, int]],
                 dim_tokens_enc: Optional[int] = None,
                 dim_tokens: int = 256,
                 depth: int = 0,
                 learnable_pos_emb: int = False,
                 image_size: Union[int, Tuple[int]] = 224,
                 mlp_ratio: int = 4.0,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 use_task_queries: bool = True,
                 task: Optional[str] = None,
                 context_tasks: Optional[list] = None,
                 use_xattn: bool = True
                 ):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens_enc = dim_tokens_enc
        self.dim_tokens = dim_tokens
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.use_task_queries = use_task_queries
        self.task = task
        self.use_xattn = use_xattn

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)

        if context_tasks is not None:
            self.task_embeddings = nn.ParameterDict(
                {task: nn.Parameter(torch.zeros(1, 1, self.dim_tokens_enc)) for task in context_tasks})
            for embedding in self.task_embeddings.values():
                trunc_normal_(embedding, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dim_tokens_enc))

        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if not self.learnable_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens_enc)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=False)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, h_posemb, w_posemb, self.dim_tokens_enc))
            trunc_normal_(self.pos_emb, std=0.02)

        # One cross attention layer followed by MLP block, an optional transformer, and an output projection
        if self.use_xattn:
            self.decoder = CrossAttention(
                dim=self.dim_tokens_enc, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate, proj_drop=drop_rate)
            self.context_norm = norm_layer(self.dim_tokens_enc)
            self.query_norm = norm_layer(self.dim_tokens_enc)
            self.out_norm = norm_layer(self.dim_tokens_enc)

            mlp_hidden_dim = int(self.dim_tokens_enc * mlp_ratio)
            self.mlp = Mlp(in_features=self.dim_tokens_enc, hidden_features=mlp_hidden_dim)

        # Optional full self-attention transformer layers
        if depth > 0:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            self.decoder_transformer = nn.Sequential(*[
                Block(dim=self.dim_tokens_enc, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                      attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
        else:
            self.decoder_transformer = nn.Identity()

        self.dim_patch = self.num_channels * self.P_H * self.P_W
        self.out_proj = nn.Linear(self.dim_tokens_enc, self.dim_patch)

        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

    def init(self, dim_tokens_enc: int = 768):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.dim_tokens_enc = dim_tokens_enc

        # Projection of encoder tokens to the patch dimension
        self.proj_context = nn.Linear(self.dim_tokens_enc, self.dim_tokens_enc)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb', 'mask_token', 'task_embeddings'}

    def generate_context_embeddings(self, input_info,
                                    bs: int,
                                    size: Tuple[int, int],
                                    device: Optional[torch.device] = None):
        context_embeddings = []
        for task, info in input_info["tasks"].items():
            if self.task_embeddings is not None and task in self.task_embeddings:
                task_emb = repeat(self.task_embeddings[task], '() () d -> b n d', b=bs, n=info['num_tokens'])
            else:
                task_emb = torch.zeros((bs, info['num_tokens'], self.dim_tokens_enc), device=device)

            if info['has_2d_posemb']:
                pos_emb = F.interpolate(self.pos_emb, size=size, mode='bilinear', align_corners=False)
                pos_emb = rearrange(pos_emb, 'b d nh nw -> b (nh nw) d')
                assert info['num_tokens'] == pos_emb.shape[1]
                task_emb = task_emb + pos_emb

            context_embeddings.append(task_emb)

        context_embeddings = torch.cat(context_embeddings, dim=1)

        return context_embeddings

    def get_queries_and_context(self, context_tokens, input_info, ids_keep, ids_restore):
        B = context_tokens.shape[0]
        H, W = input_info['image_size']
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        if 'num_global_tokens' in input_info:
            context_tokens_without_global = context_tokens[:, :-input_info['num_global_tokens']]
        else:
            context_tokens_without_global = context_tokens

        # Add mask tokens
        mask_tokens = repeat(self.mask_token, '() () d -> b n d', b=B,
                             n=input_info['num_task_tokens'] - context_tokens_without_global.shape[1])
        context_with_mask = torch.cat([context_tokens_without_global, mask_tokens], dim=1)

        # Unshuffle context_with_mask
        context_with_mask = torch.gather(context_with_mask, dim=1,
                                         index=ids_restore.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))

        # Generate context_emb and add them to context
        context_emb = self.generate_context_embeddings(input_info=input_info, bs=B, size=(N_H, N_W),
                                                       device=context_tokens.device)
        context_with_mask = context_with_mask + context_emb

        # Generate queries
        if self.use_task_queries and self.task in input_info['tasks']:
            start_idx = input_info['tasks'][self.task]['start_idx']
            end_idx = input_info['tasks'][self.task]['end_idx']
            queries = context_with_mask[:, start_idx:end_idx]
        else:
            queries = repeat(self.mask_token, '() () d -> b n d', b=B, n=N_H * N_W)
            queries_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bilinear', align_corners=False)
            queries_pos_emb = rearrange(queries_pos_emb, 'b d nh nw -> b (nh nw) d')
            queries = queries + queries_pos_emb
            if self.task_embeddings is not None and self.task in self.task_embeddings:
                queries_task_emb = repeat(self.task_embeddings[self.task], '() () d -> b n d', b=B, n=N_H * N_W)
                queries = queries + queries_task_emb

        # Unshuffle context and keep only initial context (yes, again)
        context_tokens_without_global = torch.gather(context_with_mask, dim=1,
                                                     index=ids_keep.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))

        # Add back global tokens
        if 'num_global_tokens' in input_info:
            context_tokens = torch.cat(
                [context_tokens_without_global, context_tokens[:, -input_info['num_global_tokens']:]], dim=1)
        else:
            context_tokens = context_tokens_without_global

        return queries, context_tokens

    def forward(self,
                encoder_tokens: torch.Tensor,
                input_info: Dict,
                ids_keep: torch.Tensor,
                ids_restore: torch.Tensor,
                ):
        """
        Forward pass taking output tokens from encoder and optionally a subset of them corresponding
        to this output adapter's task (needs an additional mask describing position of these tokens in the queries).

        :param encoder_tokens: Output of encoder
        :param input_info: Dictionary with information about the input modalities
        :param ids_keep: IDs of unmasked tokens (tokens given to the encoder)
        :param ids_restore: IDs to unshuffle tokens
        """
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        H, W = input_info['image_size']
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Project encoder tokens to decoder tokens
        context_tokens = self.proj_context(encoder_tokens)

        # Get queries and context
        queries, context_tokens = self.get_queries_and_context(context_tokens, input_info, ids_keep, ids_restore)

        # Perform cross attention of queries to context tokens, followed by an MLP
        if self.use_xattn:
            x = self.decoder(self.query_norm(queries), self.context_norm(context_tokens))
            x = x + self.mlp(self.out_norm(x))
        else:
            x = queries

        # Optional transformer layers if depth > 0
        x = self.decoder_transformer(x)

        # Project each token to (C * P_H * P_W)
        x = self.out_proj(x)

        # Reshape sequence of patches into image
        x = rearrange(
            x, 'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)',
            nh=N_H, nw=N_W, ph=self.P_H, pw=self.P_W, c=self.num_channels
        )

        return x


class LinearOutputAdapter(nn.Module):
    """
    Linear output adapter.

    :param num_classes: Number of classes
    :param dim_tokens_enc: Dimension of tokens from the encoder
    :param use_mean_pooling: When set to True, uses mean pooling before linear classification head.
        Otherwise, use last token (usually the global token)
    :param norm_layer: Normalization layer
    :param init_scale: Initialization scale for linear classification head
    """

    def __init__(self,
                 num_classes: int,
                 dim_tokens_enc: Optional[int] = None,
                 use_mean_pooling: bool = True,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 init_scale: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.dim_tokens_enc = dim_tokens_enc
        self.use_mean_pooling = use_mean_pooling
        self.norm_layer = norm_layer
        self.init_scale = init_scale

        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

    def init(self, dim_tokens_enc: int = 768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        self.dim_tokens_enc = dim_tokens_enc

        self.norm = self.norm_layer(self.dim_tokens_enc)
        self.head = nn.Linear(dim_tokens_enc, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.head.weight.data.mul_(self.init_scale)
        self.head.bias.data.mul_(self.init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.init(dim_tokens_enc=self.dim_tokens_enc)

    def forward(self,
                encoder_tokens: torch.Tensor,
                **kwargs):

        if self.use_mean_pooling:
            x = encoder_tokens.mean(1)
        else:
            # Global token is added at the end
            x = encoder_tokens[:, -1]

        x = self.head(self.norm(x))
        return x

class SegmenterMaskTransformerAdapter(nn.Module):
    """Output adapter inspired by the Segmenter-Mask architecture

    This head is the implementation of `Segmenter:　<https://arxiv.org/abs/2105.05633>`_.

    :param num_classes: Number of classes
    :param depth: Depth of decoder
    :param num_heads: Number of attention heads
    :param embed_dim: Dimension of decoder tokens
    :param mlp_ratio: MLP hidden dim ratio
    :param drop_path_rate: DropPath drop rate
    :param drop_rate: Dropout after MLPs and Attention
    :param attn_drop_rate: Attention matrix drop rate
    :param qkv_bias: Set to False to disable bias
    :param main_tasks: Tasks to use for the adapter. Only tokens coming from these tasks are kept.
    :param patch_size: Size of patches
    :param norm_layer: Type of normalization layer
    """

    def __init__(
            self,
            num_classes,
            depth: int = 2,
            num_heads: int = 12,
            embed_dim: int = 768,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            qkv_bias=True,
            main_tasks: str = ('rgb',),
            patch_size: int = 16,
            norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
    ):
        super().__init__()
        self.main_tasks = main_tasks
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = 40

        self.cls_emb = nn.Parameter(torch.zeros(1, num_classes, embed_dim))
        trunc_normal_(self.cls_emb, std=0.02)

        self.patch_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.classes_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,use_prompt_mask=None,prompt_size=0)
            for i in range(depth)
        ])

        self.decoder_norm = norm_layer(embed_dim)
        self.mask_norm = norm_layer(num_classes)
        self.apply(self._init_weights)

    def init(self, dim_tokens_enc: int = 768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        self.in_channels = dim_tokens_enc * len(self.main_tasks)

        # Projection of encoder tokens to the patch dimension
        self.proj_dec = nn.Linear(self.in_channels, self.embed_dim)
        self._init_weights(self.proj_dec)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        
        x = encoder_tokens
        return x 

    def forward(self, encoder_tokens: torch.Tensor, input_info: Dict):
        H, W = input_info['image_size']
        N_H, N_W = H // self.patch_size, W // self.patch_size

        x = self.adapt_tokens(encoder_tokens, input_info)
        # x = x[:,1:,:] 
        x = x[:,1 + N_H*N_W:,:] #IF RGB-D
        
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_emb), 1)

        for blk in self.blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        patches = self.patch_proj(x[:, :-self.num_classes])
        cls_seg_feat = self.classes_proj(x[:, -self.num_classes:])

        patches = F.normalize(patches, dim=2, p=2)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W)

        # Interpolate to semseg res
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")

        return masks
    
class ViTAdapter(nn.Module):

    def __init__(
            self,
            num_classes,
            depth: int = 2,
            num_heads: int = 12,
            embed_dim: int = 768,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            qkv_bias=True,
            main_tasks: str = ('rgb',),
            patch_size: int = 16,
            preds_per_patch : int = 16,
            task_specific_prompt_length : int = 200,
            norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
    ):
        super().__init__()
        self.main_tasks = main_tasks
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.task_specific_prompt_length = task_specific_prompt_length
        self.class_dim = int(embed_dim // preds_per_patch)
        self.preds_per_patch = preds_per_patch
        self.norm_mask = nn.LayerNorm(self.task_specific_prompt_length)
        self.norm = nn.LayerNorm(768)
        self.norm3 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm([144,144])
        self.batch_norm = nn.BatchNorm2d(self.num_classes)
        self.ca_1 = CrossAttention(dim=768)
        self.ca_2 = CrossAttention(dim=768)
        self.final_layer = nn.Conv2d(self.embed_dim ,self.num_classes, 1)
        self.final_prompt = nn.Conv2d(self.task_specific_prompt_length ,self.num_classes, 1)
        # self.final_prompt_2 = nn.Conv2d(self.task_specific_prompt_length ,1, 1)
        self.apply(self._init_weights)
        

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,use_prompt_mask=None,prompt_size=0)
            for i in range(depth)
        ])

        self.decoder_norm = norm_layer(embed_dim)
        self.mask_norm = norm_layer(num_classes)
        self.apply(self._init_weights)

    def init(self, dim_tokens_enc: int = 768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        self.in_channels = dim_tokens_enc * len(self.main_tasks)

        # Projection of encoder tokens to the patch dimension
        self.proj_dec = nn.Linear(self.in_channels, self.embed_dim)
        self._init_weights(self.proj_dec)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        
        x = encoder_tokens
        return x 

    def forward(self, encoder_tokens: torch.Tensor, input_info: Dict):
        H, W = input_info['image_size']
        N_H, N_W = H // self.patch_size, W // self.patch_size

        x = self.adapt_tokens(encoder_tokens, input_info)
        origin_prompts_1 = x[:,:self.task_specific_prompt_length,:]
        x = x[:,self.task_specific_prompt_length:,:]
        origin_x = x    
        x = self.proj_dec(x)   
        x = x[:,1+  self.task_specific_prompt_length +N_W:,:] #RGB만 남겨놓고
       
    #   # pseudo semseg
        p_to_im =  (self.ca_1(origin_x[:,1+  self.task_specific_prompt_length  :,:] ,origin_prompts_1)) # B image 768
        im_to_p = (origin_prompts_1 + self.ca_2(origin_prompts_1 , p_to_im)) # B prompt 768
        prompt_seg = (p_to_im @ im_to_p.transpose(1,2)) 
        prompt_seg = rearrange(prompt_seg ,"b (h w) c -> b c h w" , h = N_H , w = N_W)
        prompt_seg = self.final_prompt(prompt_seg)
        prompt_seg = self.batch_norm(prompt_seg)
        prompt_seg = F.interpolate(prompt_seg, size=(H,W) , mode= 'bilinear')       
        

        for blk in self.blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        x = rearrange(x,"b i c -> b c i")
        x = rearrange(x, "b c (nh nw) -> b c (nh) (nw)",
                        nh=N_H, nw=N_W)

        x = self.final_layer(x) # B 40 160 160  , prompt = B 40 6144
        x = F.interpolate(x, size=(H, W), mode='bilinear' )

        return x ,prompt_seg



class CrossMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 쿼리, 키, 값에 대한 프로젝션을 위한 선형 레이어
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)

        # 최종 출력을 위한 선형 레이어
        self.final_linear = nn.Linear(embed_dim, embed_dim)
        
        # 레이어 정규화
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query_seq, key_value_seq ,prompt_size):
        batch_size = query_seq.size(0)
        
        # 쿼리, 키, 값 프로젝션
        query = self.query_projection(query_seq).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_projection(key_value_seq).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_projection(key_value_seq).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        
        # 스케일드 닷 프로덕트 어텐션
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        #mask = torch.zeros_like(scores)
        #mask[:, :, prompt_size:, :prompt_size] = float("-1e4")
        
        scores = scores #+ mask
        attn = F.softmax(scores, dim=-1)

        # 어텐션 적용
        context = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # 최종 선형 레이어 및 Add & Norm
        output = self.final_linear(context)
        output = self.layer_norm(output + query_seq)
        
        return output
  
  
class MFA(nn.Module):  
    def __init__(self):
        super(MFA, self).__init__()
        self.dim_tokens_enc = 768
        self.num_classes = 40
        self.prompt_emb = nn.Linear(self.dim_tokens_enc, self.dim_tokens_enc)
        self.prompt_down = nn.Linear(self.dim_tokens_enc, self.dim_tokens_enc // 4)
        self.image_down = nn.Linear(self.dim_tokens_enc, self.dim_tokens_enc // 4)
        self.cross_attn = CrossAttention(dim=self.dim_tokens_enc // 4)
        self.conv = nn.Conv2d(self.dim_tokens_enc//4 ,self.num_classes, 1)
        self.norm = nn.LayerNorm(self.dim_tokens_enc//4)
        
    def forward(self, x, task_specific_prompt_length = 200):  # forward 메소드 정의
        self.task_specific_prompt_length = task_specific_prompt_length
        prompt = x[:, :self.task_specific_prompt_length, :]
        prompt = self.prompt_emb(prompt)
        prompt = self.prompt_down(prompt)
        image = x[:, self.task_specific_prompt_length:, :]
        image = self.image_down(image)
        x = self.cross_attn(image, prompt) # B image dim
        x = self.norm(x)
        x = rearrange(x,"b (h w) c -> b c h w" , h = 36, w = 36)
        x = self.conv(x)
        return x
 
class SELayer(nn.Module):
    def __init__(self, channel=40, reduction=16):
        super(SELayer, self).__init__()
        # Squeeze 단계: 전역 평균 풀링을 사용하여 채널별 평균을 계산
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation 단계: 두 개의 완전 연결(FC) 레이어를 사용하여 채널별 가중치를 학습
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # 채널별 중요도 가중치를 원본 입력 x에 곱함
        return y.expand_as(x)
    
class LearnableMasking(nn.Module):
    def __init__(self, channels=40, height=36, width=36):
        super(LearnableMasking, self).__init__()
        self.mask = nn.Parameter(torch.randn(channels, height, width))
    
    def forward(self, x):
        # 시그모이드 함수를 적용하여 마스크 값을 [0, 1] 범위로 조정
        mask = torch.sigmoid(self.mask)
        # 마스크 적용
        return x * mask



class CrossMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 쿼리, 키, 값에 대한 프로젝션을 위한 선형 레이어
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)

        # 최종 출력을 위한 선형 레이어
        self.final_linear = nn.Linear(embed_dim, embed_dim)
        
        # 레이어 정규화
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query_seq, key_value_seq ,prompt_size):
        batch_size = query_seq.size(0)
        
        # 쿼리, 키, 값 프로젝션
        query = self.query_projection(query_seq).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_projection(key_value_seq).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_projection(key_value_seq).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        
        # 스케일드 닷 프로덕트 어텐션
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        #mask = torch.zeros_like(scores)
        #mask[:, :, prompt_size:, :prompt_size] = float("-1e4")
        
        scores = scores #+ mask
        attn = F.softmax(scores, dim=-1)

        # 어텐션 적용
        context = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # 최종 선형 레이어 및 Add & Norm
        output = self.final_linear(context)
        output = self.layer_norm(output + query_seq)
        
        return output
  
  
class MFA(nn.Module):  
    def __init__(self):
        super(MFA, self).__init__()
        self.dim_tokens_enc = 768
        self.num_classes = 40
        self.prompt_emb = nn.Linear(self.dim_tokens_enc, self.dim_tokens_enc)
        self.prompt_down = nn.Linear(self.dim_tokens_enc, self.dim_tokens_enc // 4)
        self.image_down = nn.Linear(self.dim_tokens_enc, self.dim_tokens_enc // 4)
        self.cross_attn = CrossAttention(dim=self.dim_tokens_enc // 4)
        self.conv = nn.Conv2d(self.dim_tokens_enc//4 ,self.num_classes, 1)
        self.norm = nn.LayerNorm(self.dim_tokens_enc//4)
        
    def forward(self, x, task_specific_prompt_length = 200):  # forward 메소드 정의
        self.task_specific_prompt_length = task_specific_prompt_length
        prompt = x[:, :self.task_specific_prompt_length, :]
        prompt = self.prompt_emb(prompt)
        prompt = self.prompt_down(prompt)
        image = x[:, self.task_specific_prompt_length:, :]
        image = self.image_down(image)
        x = self.cross_attn(image, prompt) # B image dim
        x = self.norm(x)
        x = rearrange(x,"b (h w) c -> b c h w" , h = 36, w = 36)
        x = self.conv(x)
        return x
 
class SELayer(nn.Module):
    def __init__(self, channel=40, reduction=16):
        super(SELayer, self).__init__()
        # Squeeze 단계: 전역 평균 풀링을 사용하여 채널별 평균을 계산
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation 단계: 두 개의 완전 연결(FC) 레이어를 사용하여 채널별 가중치를 학습
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # 채널별 중요도 가중치를 원본 입력 x에 곱함
        return y.expand_as(x)
    
class LearnableMasking(nn.Module):
    def __init__(self, channels=40, height=36, width=36):
        super(LearnableMasking, self).__init__()
        self.mask = nn.Parameter(torch.randn(channels, height, width))
    
    def forward(self, x):
        # 시그모이드 함수를 적용하여 마스크 값을 [0, 1] 범위로 조정
        mask = torch.sigmoid(self.mask)
        # 마스크 적용
        return x * mask
             
class VPTAdapter(nn.Module):
    """Output adapter with ConvNext blocks for semantic segmentation

    :param num_classes: Number of classes
    :param num_heads: Number of attention heads
    :param embed_dim: Token dimension after projection, and before reshaping operation.
    :param preds_per_patch: Increases size of feature map by reshaping each patch  Each patch gets reshaped
        from embed_dim x 1 x 1 to (embed_dim / preds_per_patch) x (preds_per_patch ** 0.5) x (preds_per_patch ** 0.5)
    :param main_tasks: Tasks to use for the adapter. Only tokens coming from these tasks are kept.
    :param patch_size: Size of patches
    :param depth: Number of ConvNeXt blocks
    :interpolate_mode: Interpolation mode for final upsampling
    """

    def __init__(
            self,
            num_classes,
            prompt_pool : bool , 
            prompt_deep : bool , 
            not_self_attn : bool, 
            task_specific_prompt_length : int, 
            top_k : int ,
            prompt_length :int ,
            embed_dim: int = 6144,
            preds_per_patch: int = 16,
            main_tasks: Iterable[str] = ('rgb',),
            patch_size: int = 16,
            depth: int = 4,
            dim_tokens_enc : int = 768,
            interpolate_mode: str = 'bilinear',
            **kwargs,
    ):
        super().__init__()

        self.task_specific_prompt_length = task_specific_prompt_length
        self.prompt_length = prompt_length
        self.top_k = top_k
        self.dim_tokens_enc = dim_tokens_enc
        self.prompt_pool = prompt_pool
        self.prompt_deep = prompt_deep
        self.main_tasks = main_tasks
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.preds_per_patch = preds_per_patch
        self.class_dim = int(embed_dim // preds_per_patch)
        self.num_classes = int(num_classes)
        self.interpolate_mode = interpolate_mode
        self.not_self_attn = not_self_attn
        self.scale = self.dim_tokens_enc ** -1
        
        #For attention (prompt pool + task specific prompt)
        # self.self_attention1 = CrossMultiHeadAttention(embed_dim= self.dim_tokens_enc , num_heads= 8 )
     
        # self.decoder_block = DecoderBlock( dim=self.dim_tokens_enc,num_heads=8)

        
        self.norm_mask = nn.LayerNorm(self.task_specific_prompt_length)
        self.norm = nn.LayerNorm(768)
        self.norm3 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm([144,144])
        self.batch_norm = nn.BatchNorm2d(self.num_classes)
        #blocks
        
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(dim=self.class_dim )
            for _ in range(depth)
        ])
        #self.ca_1 = CrossAttention(dim=768)
        #self.ca_2 = CrossAttention(dim=768)
        self.final_layer = nn.Conv2d(self.class_dim ,self.num_classes, 1)
        #self.final_prompt = nn.Conv2d(self.task_specific_prompt_length ,self.num_classes, 1)
        # self.final_prompt_2 = nn.Conv2d(self.task_specific_prompt_length ,1, 1)
        self.apply(self._init_weights)
        
    def init(self, dim_tokens_enc: int = 768 
         ):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        self.in_channels = dim_tokens_enc * len(self.main_tasks)

        # Projection of encoder tokens to the patch dimension
        self.proj_dec = nn.Linear(self.in_channels, self.embed_dim,bias = False)
        self._init_weights(self.proj_dec)
        
        # self.proj_patch = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # self.proj_classes = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # # Task specific prompts
        # self.task_specific_prompts = nn.Parameter(torch.rand(1,self.task_specific_prompt_length
        # ,dim_tokens_enc))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        
        x = encoder_tokens
        return x 
    

    def forward(self, encoder_tokens: torch.Tensor, input_info: Dict):

        H, W = input_info['image_size']
        N_H, N_W = H // self.patch_size, W // self.patch_size
        
        x = self.adapt_tokens(encoder_tokens, input_info)
        x = x[:,1 + 2* self.task_specific_prompt_length + N_H*N_W:,:]
        x = self.proj_dec(x)
###############################################################################################################################
#EXTRA MODULE
        
    #     # 2nd path
    #     origin_prompts_1 = x[:,:self.task_specific_prompt_length,:]
    #     x = x[:,self.task_specific_prompt_length:,:] #original prompt 때고
    #     origin_x = x
    #     x = self.proj_dec(x)
        
    #     x = x[:,1+ self.task_specific_prompt_length :,:] #RGB만 남겨놓고
       
    # #   # pseudo semseg
    #     p_to_im =  (self.ca_1(origin_x[:,1+  self.task_specific_prompt_length:,:] ,origin_prompts_1)) # B image 768
    #     im_to_p = (origin_prompts_1 + self.ca_2(origin_prompts_1 , p_to_im)) # B prompt 768
    #     prompt_seg = (p_to_im @ im_to_p.transpose(1,2)) 
    #     prompt_seg = rearrange(prompt_seg ,"b (h w) c -> b c h w" , h = N_H , w = N_W)
    #     prompt_seg = self.final_prompt(prompt_seg)
    #     prompt_seg = self.batch_norm(prompt_seg)
    #     prompt_seg = F.interpolate(prompt_seg, size=(H,W) , mode= self.interpolate_mode)
        
 ################################################################################################################################
        # Real pred
        x = rearrange(x, "b n (p c) -> b (n p) c", n=N_H * N_W, p=self.preds_per_patch, c=self.class_dim) # b 160*160 384

        x = rearrange(x, "b (nh nw ph pw) c -> b c (nh ph) (nw pw)",
                        nh=N_H, nw=N_W,
                        ph=int(self.preds_per_patch ** 0.5),
                        pw=int(self.preds_per_patch ** 0.5))
        
        x = self.blocks(x)
        # x = self.norm2(x)
        x = self.final_layer(x) # B 40 160 160  , prompt = B 40 6144
        x = F.interpolate(x, size=(H, W), mode=self.interpolate_mode )
        return x 


class MaskFormer(nn.Module):
    """Output adapter with ConvNext blocks for semantic segmentation

    :param num_classes: Number of classes
    :param num_heads: Number of attention heads
    :param embed_dim: Token dimension after projection, and before reshaping operation.
    :param preds_per_patch: Increases size of feature map by reshaping each patch  Each patch gets reshaped
        from embed_dim x 1 x 1 to (embed_dim / preds_per_patch) x (preds_per_patch ** 0.5) x (preds_per_patch ** 0.5)
    :param main_tasks: Tasks to use for the adapter. Only tokens coming from these tasks are kept.
    :param patch_size: Size of patches
    :param depth: Number of ConvNeXt blocks
    :interpolate_mode: Interpolation mode for final upsampling
    """

    def __init__(
            self,
            num_classes,
            prompt_pool : bool , 
            prompt_deep : bool , 
            not_self_attn : bool, 
            task_specific_prompt_length : int, 
            top_k : int ,
            prompt_length :int ,
            embed_dim: int = 6144,
            preds_per_patch: int = 16,
            main_tasks: Iterable[str] = ('rgb',),
            patch_size: int = 16,
            depth: int = 4,
            dim_tokens_enc : int = 768,
            interpolate_mode: str = 'bilinear',
            **kwargs,
    ):
        super().__init__()

        self.task_specific_prompt_length = task_specific_prompt_length
        self.prompt_length = prompt_length
        self.top_k = top_k
        self.dim_tokens_enc = dim_tokens_enc
        self.prompt_pool = prompt_pool
        self.prompt_deep = prompt_deep
        self.main_tasks = main_tasks
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.preds_per_patch = preds_per_patch
        self.class_dim = int(embed_dim // preds_per_patch)
        self.num_classes = int(num_classes)
        self.interpolate_mode = interpolate_mode
        self.not_self_attn = not_self_attn
        self.scale = self.dim_tokens_enc ** -1
        
        #For attention (prompt pool + task specific prompt)
        # self.self_attention1 = CrossMultiHeadAttention(embed_dim= self.dim_tokens_enc , num_heads= 8 )
     
        # self.decoder_block = DecoderBlock( dim=self.dim_tokens_enc,num_heads=8)

        
        self.norm_mask = nn.LayerNorm(self.task_specific_prompt_length)
        self.norm = nn.LayerNorm(1600)
        self.norm3 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm([144,144])
        self.batch_norm = nn.BatchNorm2d(self.num_classes)
        #blocks
        dpr = [x.item() for x in torch.linspace(0, 0.1, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=12, drop_path = dpr[i] ,use_prompt_mask=None,prompt_size=0)
            for i in range(depth)
        ])
        
        self.token_blocks = CrossAttention(dim=embed_dim)
        self.cls_emb = nn.Parameter(torch.zeros(1, 40, embed_dim))
        trunc_normal_(self.cls_emb, std=0.02)
        
        
        # self.ca_1 = CrossAttention(dim=768)
        # self.ca_2 = CrossAttention(dim=768)
        self.final_layer = nn.Conv2d(self.class_dim ,self.num_classes, 1)
        # self.final_prompt = nn.Conv2d(self.task_specific_prompt_length ,self.num_classes, 1)
        # self.final_prompt_2 = nn.Conv2d(self.task_specific_prompt_length ,1, 1)
        self.apply(self._init_weights)
        
    def init(self, dim_tokens_enc: int = 768 
         ):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        self.in_channels = dim_tokens_enc * len(self.main_tasks)

        # Projection of encoder tokens to the patch dimension
        self.proj_dec = nn.Linear(self.in_channels, self.embed_dim,bias = False)
        self._init_weights(self.proj_dec)
        
        # self.proj_patch = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # self.proj_classes = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # # Task specific prompts
        # self.task_specific_prompts = nn.Parameter(torch.rand(1,self.task_specific_prompt_length
        # ,dim_tokens_enc))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        
        x = encoder_tokens
        return x 
    

    def forward(self, encoder_tokens: torch.Tensor, input_info: Dict):

        H, W = input_info['image_size']
        N_H, N_W = H // self.patch_size, W // self.patch_size
        
        x = self.adapt_tokens(encoder_tokens, input_info)
        cls_emb = self.cls_emb.expand(x.shape[0], -1, -1)
        token = self.token_blocks(cls_emb , x) # 40, 768
        x = self.proj_dec(x)
        
        x = x[:,1 +N_H*N_W:,:] #RGB만 남겨놓고
 ################################################################################################################################
        # Real pred
        for blk in self.blocks:
            x = blk(x)
        x = rearrange(x, "b n c -> b c n") # b 160*160 384
        x = token @ x
        x = self.norm(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=40, w=40)
        x = F.interpolate(x, size=(H, W), mode=self.interpolate_mode )

        return x 
    
    
class ConvNeXtAdapter(nn.Module):
    """Output adapter with ConvNext blocks for semantic segmentation

    :param num_classes: Number of classes
    :param num_heads: Number of attention heads
    :param embed_dim: Token dimension after projection, and before reshaping operation.
    :param preds_per_patch: Increases size of feature map by reshaping each patch  Each patch gets reshaped
        from embed_dim x 1 x 1 to (embed_dim / preds_per_patch) x (preds_per_patch ** 0.5) x (preds_per_patch ** 0.5)
    :param main_tasks: Tasks to use for the adapter. Only tokens coming from these tasks are kept.
    :param patch_size: Size of patches
    :param depth: Number of ConvNeXt blocks
    :interpolate_mode: Interpolation mode for final upsampling
    """

    def __init__(
            self,
            num_classes,
            prompt_pool : bool , 
            prompt_deep : bool , 
            not_self_attn : bool, 
            task_specific_prompt_length : int, 
            top_k : int ,
            prompt_length :int ,
            embed_dim: int = 6144,
            preds_per_patch: int = 16,
            main_tasks: Iterable[str] = ('rgb',),
            patch_size: int = 16,
            depth: int = 4,
            dim_tokens_enc : int = 768,
            interpolate_mode: str = 'bilinear',
            **kwargs,
    ):
        super().__init__()

        self.task_specific_prompt_length = task_specific_prompt_length
        self.prompt_length = prompt_length
        self.top_k = top_k
        self.dim_tokens_enc = dim_tokens_enc
        self.prompt_pool = prompt_pool
        self.prompt_deep = prompt_deep
        self.main_tasks = main_tasks
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.preds_per_patch = preds_per_patch
        self.class_dim = int(embed_dim // preds_per_patch)
        self.num_classes = int(num_classes)
        self.interpolate_mode = interpolate_mode
        self.not_self_attn = not_self_attn
        self.scale = self.dim_tokens_enc ** -1
        
        #For attention (prompt pool + task specific prompt)
        # self.self_attention1 = CrossMultiHeadAttention(embed_dim= self.dim_tokens_enc , num_heads= 8 )
     
        # self.decoder_block = DecoderBlock( dim=self.dim_tokens_enc,num_heads=8)

        
        self.norm_mask = nn.LayerNorm(self.task_specific_prompt_length)
        self.norm = nn.LayerNorm(768)
        self.norm3 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm([144,144])
        self.batch_norm = nn.BatchNorm2d(self.num_classes)
        #blocks
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(dim=self.class_dim )
            for _ in range(depth)
        ])
        self.ca_1 = CrossAttention(dim=768)
        self.ca_2 = CrossAttention(dim=768)
        self.final_layer = nn.Conv2d(self.class_dim ,self.num_classes, 1)
        self.final_prompt = nn.Conv2d(self.task_specific_prompt_length ,self.num_classes, 1)
        # self.final_prompt_2 = nn.Conv2d(self.task_specific_prompt_length ,1, 1)
        self.apply(self._init_weights)
        
    def init(self, dim_tokens_enc: int = 768 
         ):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        self.in_channels = dim_tokens_enc * len(self.main_tasks)

        # Projection of encoder tokens to the patch dimension
        self.proj_dec = nn.Linear(self.in_channels, self.embed_dim,bias = False)
        self._init_weights(self.proj_dec)
        
        # self.proj_patch = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # self.proj_classes = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # # Task specific prompts
        # self.task_specific_prompts = nn.Parameter(torch.rand(1,self.task_specific_prompt_length
        # ,dim_tokens_enc))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        
        x = encoder_tokens
        return x 
    

    def forward(self, encoder_tokens: torch.Tensor, input_info: Dict):

        H, W = input_info['image_size']
        N_H, N_W = H // self.patch_size, W // self.patch_size
        
        x = self.adapt_tokens(encoder_tokens, input_info)
###############################################################################################################################
#EXTRA MODULE
        
        # 2nd path
        origin_prompts_1 = x[:,:self.task_specific_prompt_length,:]
        x = x[:,self.task_specific_prompt_length:,:] #original prompt 때고
        origin_x = x
        x = self.proj_dec(x)
        
        x = x[:,1+ self.task_specific_prompt_length:,:] #RGB만 남겨놓고
       
    #   # pseudo semseg
        p_to_im =  (self.ca_1(origin_x[:,1+  self.task_specific_prompt_length : ,:] ,origin_prompts_1)) # B image 768
        im_to_p = (origin_prompts_1 + self.ca_2(origin_prompts_1 , p_to_im)) # B prompt 768
        prompt_seg = (p_to_im @ im_to_p.transpose(1,2)) 
        prompt_seg = rearrange(prompt_seg ,"b (h w) c -> b c h w" , h = N_H , w = N_W)
        prompt_seg = self.final_prompt(prompt_seg)
        prompt_seg = self.batch_norm(prompt_seg)
        prompt_seg = F.interpolate(prompt_seg, size=(H,W) , mode= self.interpolate_mode)
        
 ################################################################################################################################
        # Real pred
        x = rearrange(x, "b n (p c) -> b (n p) c", n=N_H * N_W, p=self.preds_per_patch, c=self.class_dim) # b 160*160 384

        x = rearrange(x, "b (nh nw ph pw) c -> b c (nh ph) (nw pw)",
                        nh=N_H, nw=N_W,
                        ph=int(self.preds_per_patch ** 0.5),
                        pw=int(self.preds_per_patch ** 0.5))
        
        x = self.blocks(x)
        # x = self.norm2(x)
        x = self.final_layer(x) # B 40 160 160  , prompt = B 40 6144
        x = F.interpolate(x, size=(H, W), mode=self.interpolate_mode )
        return x , prompt_seg  






class CrossMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 쿼리, 키, 값에 대한 프로젝션을 위한 선형 레이어
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)

        # 최종 출력을 위한 선형 레이어
        self.final_linear = nn.Linear(embed_dim, embed_dim)
        
        # 레이어 정규화
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query_seq, key_value_seq ,prompt_size):
        batch_size = query_seq.size(0)
        
        # 쿼리, 키, 값 프로젝션
        query = self.query_projection(query_seq).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_projection(key_value_seq).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_projection(key_value_seq).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        
        # 스케일드 닷 프로덕트 어텐션
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        #mask = torch.zeros_like(scores)
        #mask[:, :, prompt_size:, :prompt_size] = float("-1e4")
        
        scores = scores #+ mask
        attn = F.softmax(scores, dim=-1)

        # 어텐션 적용
        context = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # 최종 선형 레이어 및 Add & Norm
        output = self.final_linear(context)
        output = self.layer_norm(output + query_seq)
        
        return output
  
  
class MFA(nn.Module):  
    def __init__(self):
        super(MFA, self).__init__()
        self.dim_tokens_enc = 768
        self.num_classes = 40
        self.prompt_emb = nn.Linear(self.dim_tokens_enc, self.dim_tokens_enc)
        self.prompt_down = nn.Linear(self.dim_tokens_enc, self.dim_tokens_enc // 4)
        self.image_down = nn.Linear(self.dim_tokens_enc, self.dim_tokens_enc // 4)
        self.cross_attn = CrossAttention(dim=self.dim_tokens_enc // 4)
        self.conv = nn.Conv2d(self.dim_tokens_enc//4 ,self.num_classes, 1)
        self.norm = nn.LayerNorm(self.dim_tokens_enc//4)
        
    def forward(self, x, task_specific_prompt_length = 200):  # forward 메소드 정의
        self.task_specific_prompt_length = task_specific_prompt_length
        prompt = x[:, :self.task_specific_prompt_length, :]
        prompt = self.prompt_emb(prompt)
        prompt = self.prompt_down(prompt)
        image = x[:, self.task_specific_prompt_length:, :]
        image = self.image_down(image)
        x = self.cross_attn(image, prompt) # B image dim
        x = self.norm(x)
        x = rearrange(x,"b (h w) c -> b c h w" , h = 36, w = 36)
        x = self.conv(x)
        return x
 
class SELayer(nn.Module):
    def __init__(self, channel=40, reduction=16):
        super(SELayer, self).__init__()
        # Squeeze 단계: 전역 평균 풀링을 사용하여 채널별 평균을 계산
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation 단계: 두 개의 완전 연결(FC) 레이어를 사용하여 채널별 가중치를 학습
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # 채널별 중요도 가중치를 원본 입력 x에 곱함
        return y.expand_as(x)
    
class LearnableMasking(nn.Module):
    def __init__(self, channels=40, height=36, width=36):
        super(LearnableMasking, self).__init__()
        self.mask = nn.Parameter(torch.randn(channels, height, width))
    
    def forward(self, x):
        # 시그모이드 함수를 적용하여 마스크 값을 [0, 1] 범위로 조정
        mask = torch.sigmoid(self.mask)
        # 마스크 적용
        return x * mask
             
class VPTAdapter(nn.Module):
    """Output adapter with ConvNext blocks for semantic segmentation

    :param num_classes: Number of classes
    :param num_heads: Number of attention heads
    :param embed_dim: Token dimension after projection, and before reshaping operation.
    :param preds_per_patch: Increases size of feature map by reshaping each patch  Each patch gets reshaped
        from embed_dim x 1 x 1 to (embed_dim / preds_per_patch) x (preds_per_patch ** 0.5) x (preds_per_patch ** 0.5)
    :param main_tasks: Tasks to use for the adapter. Only tokens coming from these tasks are kept.
    :param patch_size: Size of patches
    :param depth: Number of ConvNeXt blocks
    :interpolate_mode: Interpolation mode for final upsampling
    """

    def __init__(
            self,
            num_classes,
            prompt_pool : bool , 
            prompt_deep : bool , 
            not_self_attn : bool, 
            task_specific_prompt_length : int, 
            top_k : int ,
            prompt_length :int ,
            embed_dim: int = 6144,
            preds_per_patch: int = 16,
            main_tasks: Iterable[str] = ('rgb',),
            patch_size: int = 16,
            depth: int = 4,
            dim_tokens_enc : int = 768,
            interpolate_mode: str = 'bilinear',
            **kwargs,
    ):
        super().__init__()

        self.task_specific_prompt_length = task_specific_prompt_length
        self.prompt_length = prompt_length
        self.top_k = top_k
        self.dim_tokens_enc = dim_tokens_enc
        self.prompt_pool = prompt_pool
        self.prompt_deep = prompt_deep
        self.main_tasks = main_tasks
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.preds_per_patch = preds_per_patch
        self.class_dim = int(embed_dim // preds_per_patch)
        self.num_classes = int(num_classes)
        self.interpolate_mode = interpolate_mode
        self.not_self_attn = not_self_attn
        self.scale = self.dim_tokens_enc ** -1
        
        #For attention (prompt pool + task specific prompt)
        # self.self_attention1 = CrossMultiHeadAttention(embed_dim= self.dim_tokens_enc , num_heads= 8 )
     
        # self.decoder_block = DecoderBlock( dim=self.dim_tokens_enc,num_heads=8)

        
        self.norm_mask = nn.LayerNorm(self.task_specific_prompt_length)
        self.norm = nn.LayerNorm(768)
        self.norm3 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm([144,144])
        self.batch_norm = nn.BatchNorm2d(self.num_classes)
        #blocks
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(dim=self.class_dim )
            for _ in range(depth)
        ])
        #self.ca_1 = CrossAttention(dim=768)
        #self.ca_2 = CrossAttention(dim=768)
        self.final_layer = nn.Conv2d(self.class_dim ,self.num_classes, 1)
        #self.final_prompt = nn.Conv2d(self.task_specific_prompt_length ,self.num_classes, 1)
        # self.final_prompt_2 = nn.Conv2d(self.task_specific_prompt_length ,1, 1)
        self.apply(self._init_weights)
        
    def init(self, dim_tokens_enc: int = 768 
         ):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        self.in_channels = dim_tokens_enc * len(self.main_tasks)

        # Projection of encoder tokens to the patch dimension
        self.proj_dec = nn.Linear(self.in_channels, self.embed_dim,bias = False)
        self._init_weights(self.proj_dec)
        
        # self.proj_patch = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # self.proj_classes = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # # Task specific prompts
        # self.task_specific_prompts = nn.Parameter(torch.rand(1,self.task_specific_prompt_length
        # ,dim_tokens_enc))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        
        x = encoder_tokens
        return x 
    

    def forward(self, encoder_tokens: torch.Tensor, input_info: Dict):

        H, W = input_info['image_size']
        N_H, N_W = H // self.patch_size, W // self.patch_size
        
        x = self.adapt_tokens(encoder_tokens, input_info)
        x = x[:,1 + 2* self.task_specific_prompt_length:,:]
        x = self.proj_dec(x)
###############################################################################################################################
#EXTRA MODULE
        
    #     # 2nd path
    #     origin_prompts_1 = x[:,:self.task_specific_prompt_length,:]
    #     x = x[:,self.task_specific_prompt_length:,:] #original prompt 때고
    #     origin_x = x
    #     x = self.proj_dec(x)
        
    #     x = x[:,1+ self.task_specific_prompt_length :,:] #RGB만 남겨놓고
       
    # #   # pseudo semseg
    #     p_to_im =  (self.ca_1(origin_x[:,1+  self.task_specific_prompt_length:,:] ,origin_prompts_1)) # B image 768
    #     im_to_p = (origin_prompts_1 + self.ca_2(origin_prompts_1 , p_to_im)) # B prompt 768
    #     prompt_seg = (p_to_im @ im_to_p.transpose(1,2)) 
    #     prompt_seg = rearrange(prompt_seg ,"b (h w) c -> b c h w" , h = N_H , w = N_W)
    #     prompt_seg = self.final_prompt(prompt_seg)
    #     prompt_seg = self.batch_norm(prompt_seg)
    #     prompt_seg = F.interpolate(prompt_seg, size=(H,W) , mode= self.interpolate_mode)
        
 ################################################################################################################################
        # Real pred
        x = rearrange(x, "b n (p c) -> b (n p) c", n=N_H * N_W, p=self.preds_per_patch, c=self.class_dim) # b 160*160 384

        x = rearrange(x, "b (nh nw ph pw) c -> b c (nh ph) (nw pw)",
                        nh=N_H, nw=N_W,
                        ph=int(self.preds_per_patch ** 0.5),
                        pw=int(self.preds_per_patch ** 0.5))
        
        x = self.blocks(x)
        # x = self.norm2(x)
        x = self.final_layer(x) # B 40 160 160  , prompt = B 40 6144
        x = F.interpolate(x, size=(H, W), mode=self.interpolate_mode )
        return x 
    
class DPTOutputAdapter(nn.Module):
    """DPT output adapter.

    :param num_classes: Number of output channelsv
    :param stride_level: tride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param hooks: Index of intermediate layers
    :param layer_dims: Dimension of intermediate layers
    :param feature_dim: Feature dimension
    :param use_bn: If set to True, activates batch norm
    :param dim_tokens_enc:  Dimension of tokens coming from encoder
    """

    def __init__(self,
                 prompt_pool: bool,
                 prompt_shallow :bool,
                 prompt_deep : bool ,
                 prompt_length : int,
                 top_k : int,
                 pool_size : int,
                 num_classes: int = 3,
                 stride_level: int = 1,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 main_tasks: Iterable[str] = ('rgb',),
                 hooks: List[int] = [2, 5, 8, 11],
                 layer_dims: List[int] = [96, 192, 384, 768],
                 feature_dim: int = 256,
                 use_bn: bool = False,
                 dim_tokens_enc: Optional[int] = None,
                 head_type: str = 'regression',
                 **kwargs):
        super().__init__()
        
        self.prompt_pool= prompt_pool
        self.prompt_shallow = prompt_shallow
        self.prompt_deep =  prompt_deep 
        self.prompt_length =  prompt_length
        self.top_k =  top_k
        self.pool_size =  pool_size 
        self.num_channels = num_classes
        self.stride_level = stride_level
        self.patch_size = pair(patch_size)
        self.main_tasks = main_tasks
        self.hooks = hooks
        self.layer_dims = layer_dims
        self.feature_dim = feature_dim
        self.dim_tokens_enc = dim_tokens_enc * len(self.main_tasks) if dim_tokens_enc is not None else None
        self.head_type = head_type

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)

        self.scratch = make_scratch(layer_dims, feature_dim, groups=1, expand=False)

        self.scratch.refinenet1 = make_fusion_block(feature_dim, use_bn)
        self.scratch.refinenet2 = make_fusion_block(feature_dim, use_bn)
        self.scratch.refinenet3 = make_fusion_block(feature_dim, use_bn)
        self.scratch.refinenet4 = make_fusion_block(feature_dim, use_bn)

        if self.head_type == 'regression':
            # The "DPTDepthModel" head
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(feature_dim // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, self.num_channels, kernel_size=1, stride=1, padding=0)
            )
        elif self.head_type == 'semseg':
            # The "DPTSegmentationModel" head
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature_dim) if use_bn else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(feature_dim, self.num_channels, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
        else:
            raise ValueError('DPT head_type must be "regression" or "semseg".')

        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

    def init(self, dim_tokens_enc: int = 768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        self.dim_tokens_enc = dim_tokens_enc * len(self.main_tasks)

        # Set up activation postprocessing layers

        self.act_1_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc,
                out_channels=self.layer_dims[0],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[0],
                out_channels=self.layer_dims[0],
                kernel_size=4, stride=4, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        self.act_2_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc,
                out_channels=self.layer_dims[1],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[1],
                out_channels=self.layer_dims[1],
                kernel_size=2, stride=2, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        self.act_3_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc,
                out_channels=self.layer_dims[2],
                kernel_size=1, stride=1, padding=0,
            )
        )

        self.act_4_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc,
                out_channels=self.layer_dims[3],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[3],
                out_channels=self.layer_dims[3],
                kernel_size=3, stride=2, padding=1,
            )
        )

        self.act_postprocess = nn.ModuleList([
            self.act_1_postprocess,
            self.act_2_postprocess,
            self.act_3_postprocess,
            self.act_4_postprocess
        ])

    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        x = []
        for task in self.main_tasks:
            start_idx = input_info['tasks'][task]['start_idx']
            end_idx = input_info['tasks'][task]['end_idx']
            x.append(encoder_tokens[:, start_idx:end_idx])

        x = torch.cat(x, dim=-1)
        return x

    def forward(self, encoder_tokens: List[torch.Tensor], input_info: Dict):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        H, W = input_info['image_size']
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)
        
        want_length  = N_H * N_W # 900 or 1600.. 
        
        if self.prompt_pool :
            layers = []
            for hook in self.hooks :
                total_tokens = encoder_tokens[0].shape[1]
                cut = total_tokens - want_length
                # Assuming prompt_length is the length of prompt tokens for each layer
                # Define or calculate the length of the prompt
                layers.append(encoder_tokens[hook][:, cut : total_tokens])
                
        else:   
            # Hook decoder onto 4 layers from specified ViT layers
            layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l, input_info) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        # Postprocess activations
        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]

        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # Output head
        out = self.head(path_1)

        return out
