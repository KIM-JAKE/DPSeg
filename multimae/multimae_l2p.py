# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------
import matplotlib.pyplot as plt
import itertools
import math
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union

import torch
from einops import rearrange, repeat
from torch import nn
from torch.distributions.dirichlet import Dirichlet
from torch.nn import Dropout
from utils.registry import register_model

from .multimae_utils import Block, trunc_normal_

__all__ = [
    'pretrain_multimae_base',
    'pretrain_multimae_large',
    'multivit_base',
    'multivit_large',
]
from multimae.prompt import Prompt

class MultiMAE(nn.Module):
    """MultiMAE: Multi-task Multi-modal Masked Autoencoder
    This module performs masking in its forward pass.
    The MultiViT module defined below inherits from this module and performs a regular forward pass,
    and should be used instead for downstream tasks


    :param input_adapters: Dictionary of task -> input adapters
    :param output_adapters: Optional dictionary of task -> output adapters

    :param num_global_tokens: Number of additional global tokens to add (like cls tokens), default is 1
    :param dim_tokens: Dimension of encoder tokens
    :param depth: Depth of encoder
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim ratio
    :param qkv_bias: Set to False to disable bias
    :param drop_rate: Dropout after MLPs and Attention
    :param attn_drop_rate: Attention matrix drop rate
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    """
    def __init__(self,
                 input_adapters: Dict[str, nn.Module],
                 output_adapters: Optional[Dict[str, nn.Module]],
                 prompt_shallow : bool , 
                 prompt_deep : bool,
                 prompt_length : int ,
                 pool_size : int , 
                 top_k : int  , 
                 prompt_pool : bool,
                 task_specific_prompt_length : int ,
                 use_prompt_mask : bool , 
                 num_global_tokens: int = 1,
                 dim_tokens: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 prompt_dropout_rate : float = 0.0 ,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),**kwargs):
        super().__init__()

        self.use_prompt_mask = use_prompt_mask
        self.prompt_shallow = prompt_shallow
        self.prompt_deep = prompt_deep
        self.prompt_length = prompt_length
        self.pool_size = pool_size
        self.top_k = top_k
        self.prompt_pool = prompt_pool
        self.dim_tokens = dim_tokens
        self.prompt_dropout_rate = prompt_dropout_rate
        self.task_specific_prompt_length = task_specific_prompt_length
        self.prompt_proj = nn.Identity()
        
        self.prompt_emb = nn.Linear(self.task_specific_prompt_length, self.dim_tokens , bias = False)
        
        #prompt_dropout
        
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, dim_tokens))
        
        self.prompt_dropout = Dropout(self.prompt_dropout_rate)

        self.task_specific_prompts_1 = nn.Parameter(torch.rand(1, self.task_specific_prompt_length, self.dim_tokens))
        # self.task_specific_prompts_2 = nn.Parameter(torch.rand(1, self.task_specific_prompt_length, self.dim_tokens))  
        
        # # 첫 번째 프롬프트는 -1.0에서 1.0 사이의 값으로 초기화
        nn.init.uniform_(self.task_specific_prompts_1, a=-1.0, b=1.0)

        # # 두 번째 프롬프트는 -0.01에서 0.01 사이의 매우 작은 값으로 초기화
        # nn.init.uniform_(self.task_specific_prompts_2, a=-0.01, b=0.01)
        
        #learnable weight
        self.raw_parameter_seg = torch.nn.Parameter(torch.tensor(0.6))
        self.raw_parameter_depth = torch.nn.Parameter(torch.tensor(0.6))
        # self.extra_module = extra_module()
        #prompts
        # self.prompt = Prompt(length=self.prompt_length, embed_dim=self.dim_tokens, prompt_pool=self.prompt_pool,
        #                          top_k=self.top_k, pool_size=self.pool_size)
        if self.prompt_pool:
            if self.prompt_deep:
                self.layer_prompt_pools = nn.ModuleList([
                    Prompt(length=prompt_length, 
                          embed_dim=dim_tokens, 
                          pool_size=pool_size,
                          prompt_pool=True,
                          top_k=top_k)
                    for _ in range(12)  # 각 레이어에 대응하는 프롬프트 풀 초기화
                ])

        # Initialize input and output adapters
        for adapter in input_adapters.values():
            adapter.init(dim_tokens=dim_tokens,
            prompt_shallow = prompt_shallow,
            prompt_deep = prompt_deep)
        self.input_adapters = nn.ModuleDict(input_adapters)
        if output_adapters is not None:
            for adapter in output_adapters.values():
                adapter.init(dim_tokens_enc=dim_tokens,
                )
            self.output_adapters = nn.ModuleDict(output_adapters)
        else:
            self.output_adapters = None
        # Additional learnable tokens that can be used by encoder to process/store global information

        # Transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.encoder = nn.Sequential(*[
            Block(
                dim=dim_tokens,
                use_prompt_mask=
                    (False if ( 0<= i < 12 )  or ( 8 <= i < 12 ) else False),
                prompt_size=
                    (self.top_k * self.prompt_length),
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer
            )
            for i in range(depth) 
        ])
           
        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv2d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.encoder)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = {'global_tokens' , 'task_specific_prompts_1' ,'task_specific_prompts_2' , 'layer_prompt_pools' }

        for task, adapter in self.input_adapters.items():
            if hasattr(adapter, 'no_weight_decay'):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f'input_adapters.{task}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        for task, adapter in self.output_adapters.items():
            if hasattr(adapter, 'no_weight_decay'):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f'output_adapters.{task}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def sample_alphas(self, B: int, n_tasks: int, alphas: float = 1.0, eps: float = 1e-5):
        """
        Sample alphas for Dirichlet sampling such that tasks are first uniformly chosen and then Dirichlet sampling
        is performed over the chosen ones.

        :param B: Batch size
        :param n_tasks: Number of input tasks
        :param alphas: Float or list to multiply task choices {0,1} by
        :param eps: Small constant since Dirichlet alphas need to be positive
        """
        valid_task_choices = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=n_tasks)][1:])
        rand_per_sample_choice = torch.randint(0, len(valid_task_choices), (B,))
        alphas_tensor = torch.index_select(valid_task_choices, 0, rand_per_sample_choice)
        alphas_tensor = alphas_tensor * torch.tensor(alphas) + eps
        return alphas_tensor

    def generate_random_masks(self,
                            input_tokens: Dict[str, torch.Tensor],
                            num_encoded_tokens: int,
                            alphas: Union[float, List[float]] = 1.0,
                            sample_tasks_uniformly: bool = False) :
        """
        Sample a total of num_encoded_tokens from different tasks using Dirichlet sampling.

        :param input_tokens: Dictionary of tensors to sample num_encoded_tokens from
        :param num_encoded_tokens: Number of tokens to select
        :param alphas: Dirichlet distribution parameter alpha. Lower alpha = harder,
            less uniform sampling. Can be float or list of floats.
        :param sample_tasks_uniformly: Set to True to first sample 1-n_tasks uniformly at random
            for each sample in the batch. Dirichlet sampling is then done over selected subsets.
        """
        B = list(input_tokens.values())[0].shape[0]
        device = list(input_tokens.values())[0].device

        alphas = [alphas] * len(input_tokens) if isinstance(alphas, float) else alphas
        if sample_tasks_uniformly:
            alphas = self.sample_alphas(B, len(input_tokens), alphas=alphas)
            task_sampling_dist = Dirichlet(alphas).sample().to(device)
        else:
            task_sampling_dist = Dirichlet(torch.Tensor(alphas)).sample((B,)).to(device)

        samples_per_task = (task_sampling_dist * num_encoded_tokens).round().long()

        task_masks = []
        num_tokens_per_task = [task_tokens.shape[1] for task_tokens in input_tokens.values()]
        for i, num_tokens in enumerate(num_tokens_per_task):
            # Use noise to shuffle arange
            noise = torch.rand(B, num_tokens, device=device)  # noise in [0, 1]
            ids_arange_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(B, -1)
            mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)
            # 0 is keep (unmasked), 1 is remove (masked)
            mask = torch.where(mask < samples_per_task[:, i].unsqueeze(1), 0, 1)
            task_masks.append(mask)

        mask_all = torch.cat(task_masks, dim=1)
        ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_encoded_tokens]

        # Update binary mask to adjust for task rounding
        mask_all = torch.ones_like(mask_all)
        mask_all[:, :num_encoded_tokens] = 0
        # Unshuffle to get the binary mask
        mask_all = torch.gather(mask_all, dim=1, index=ids_restore)
        # Split to get task masks
        task_masks = torch.split(mask_all, num_tokens_per_task, dim=1)
        # Convert to dict
        task_masks = {domain: mask for domain, mask in zip(input_tokens.keys(), task_masks)}

        return task_masks, ids_keep, ids_restore

    @staticmethod
    def make_mask(N_H, N_W, xy_idxs, full_tasks=[], indicate_visible=True, flatten=True, device='cuda'):
        """
        Creates masks for each task, given lists of un-masked x,y coordinates.
        """
        xy_idxs = {
            k: torch.LongTensor(v)
            for k, v in xy_idxs.items()
        }

        task_masks = {
            k: torch.ones(N_H, N_W).to(device)
            for k in xy_idxs.keys()
        }

        for k in xy_idxs.keys():
            if len(xy_idxs[k]) > 0:
                task_masks[k][xy_idxs[k][:, 1], xy_idxs[k][:, 0]] = 0

        for task in full_tasks:
            task_masks[task][:] = 0

        if not indicate_visible:
            task_masks = {k: 1 - v for k, v in task_masks.items()}

        if flatten:
            task_masks = {k: v.flatten().unsqueeze(0) for k, v in task_masks.items()}

        return task_masks

    def generate_input_info(self, input_task_tokens, image_size):
        input_info = OrderedDict()
        i = 0
        input_info['tasks'] = {}

        for domain, tensor in input_task_tokens.items():
            num_tokens = tensor.shape[1]
            d = {
                'num_tokens': num_tokens,
                'has_2d_posemb': True,  # TODO: Modify when adding non-2D tasks
                'start_idx': i,
                'end_idx': i + num_tokens,
            }
            i += num_tokens
            input_info['tasks'][domain] = d

        input_info['image_size'] = image_size
        input_info['num_task_tokens'] = i
        input_info['num_global_tokens'] = self.num_global_tokens

        return input_info

    def forward(self, 
                x: Union[Dict[str, torch.Tensor], torch.Tensor], 
                mask_inputs: bool = True,
                task_masks: Dict[str, torch.Tensor] = None,
                num_encoded_tokens: int = 128,
                alphas: Union[float, List[float]] = 1.0,
                sample_tasks_uniformly: bool = False,
                fp32_output_adapters: List[str] = []):
        """
        Forward pass through input adapters, transformer encoder and output adapters.
        If specified, will randomly drop input tokens.

        :param x: Input tensor or dictionary of tensors
        :param mask_inputs: Set to True to enable random masking of input patches
        :param task_masks: Optional dictionary of task->mask pairs.
        :param num_encoded_tokens: Number of tokens to randomly select for encoder.
            Only used if mask_inputs is True.
        :param alphas: Dirichlet distribution parameter alpha for task sampling.
            Higher alpha = harder, less uniform sampling. Can be float or list of floats.
        :param sample_tasks_uniformly: Set to True if tasks should be uniformly presampled,
            before Dirichlet sampling decides share of masked tokens between them.
        :param fp32_output_adapters: List of task identifiers to force output adapters to
            run with mixed precision turned off for stability reasons.
        """

        ## Processing input modalities
        # If input x is a Tensor, assume it's RGB
        x = {'rgb': x} if isinstance(x, torch.Tensor) else x

        # Need image size for tokens->image reconstruction
        # We assume that at least one of rgb or semseg is given as input before masking
        if 'rgb' in x:
            B, C, H, W = x['rgb'].shape
        elif 'semseg' in x:
            B, H, W = x['semseg'].shape
            H *= self.input_adapters['semseg'].stride_level
            W *= self.input_adapters['semseg'].stride_level
        else:
            B, C, H, W = list(x.values())[0].shape  # TODO: Deal with case where not all have same shape

        # Encode selected inputs to tokens
        input_task_tokens = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }

        input_task_tokens = input_task_tokens['prompted_embedding']

        input_info = self.generate_input_info(input_task_tokens=input_task_tokens, image_size=(H, W))

        # Select random subset of tokens from the chosen input tasks and concatenate them
        if mask_inputs:
            num_encoded_tokens = num_encoded_tokens if num_encoded_tokens is not None else self.num_encoded_tokens
        else:
            num_encoded_tokens = sum([tensor.shape[1] for tensor in input_task_tokens.values()])

        ## Generating masks
        if task_masks is None:
            task_masks, ids_keep, ids_restore = self.generate_random_masks(
                input_task_tokens,
                num_encoded_tokens,
                alphas=alphas,
                sample_tasks_uniformly=sample_tasks_uniformly
            )
        else:
            mask_all = torch.cat([task_masks[task] for task in input_task_tokens.keys()], dim=1)
            ids_shuffle = torch.argsort(mask_all, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :(mask_all == 0).sum()]

        input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)

        # Apply mask
        input_tokens = torch.gather(input_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[2]))

        # Add global tokens to input tokens
        global_tokens = repeat(self.global_tokens, '() n d -> b n d', b=B)
        
        encoder_tokens = self.encoder(input_tokens)

        ## Output decoders
        if self.output_adapters is None:
            return encoder_tokens, task_masks

        # Decode tokens for each task using task-specific output adapters
        preds = {
            domain: self.output_adapters[domain](
                encoder_tokens=encoder_tokens,
                input_info=input_info,
                ids_keep=ids_keep,
                ids_restore=ids_restore,
            )
            for domain in self.output_adapters
            if domain not in fp32_output_adapters
        }
        # Force running selected output adapters in fp32 mode
        with torch.cuda.amp.autocast(enabled=False):
            for domain in fp32_output_adapters:
                if domain not in self.output_adapters:
                    continue
                preds[domain] = self.output_adapters[domain](
                    encoder_tokens=encoder_tokens.float(),
                    input_info=input_info,
                    ids_keep=ids_keep,
                    ids_restore=ids_restore,
                )
        
        return preds, task_masks


@register_model
def pretrain_multimae_base(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        **kwargs):
    model = MultiMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

@register_model
def pretrain_multimae_large(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        **kwargs):
    model = MultiMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

import torch.nn.functional as F

class extra_module(nn.Module):
    def __init__(self, input_dim =768, output_dim = 192):
        super(extra_module, self).__init__()
        # 1D 컨볼루션을 사용하여 차원을 줄입니다.
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=1)

    def forward(self, x):
        # (batch_size, seq_len, features) -> (batch_size, features, seq_len)으로 변경
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, scale_factor=4,  mode='linear', align_corners=False)
        return x
    
class MultiViT(MultiMAE):
    """MultiViT: Multi-modal Vision Transformer
    This is MultiMAE without masking and with a simplified / faster forward pass


    :param input_adapters: Dictionary of task -> input adapters
    :param output_adapters: Optional dictionary of task -> output adapters

    :param num_global_tokens: Number of additional global tokens to add (like cls tokens), default is 1
    :param dim_tokens: Dimension of encoder tokens
    :param depth: Depth of encoder
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim ratio
    :param qkv_bias: Set to False to disable bias
    :param drop_rate: Dropout after MLPs and Attention
    :param attn_drop_rate: Attention matrix drop rate
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    """
    def process_input(self, x):
        # If input x is a Tensor, assume it's RGB
        x = {'rgb': x} if isinstance(x, torch.Tensor) else x
        # Need image size for tokens->image reconstruction
        if 'rgb' in x:
            B, _, H, W = x['rgb'].shape
        elif 'semseg' in x:
            B, H, W = x['semseg'].shape
            H *= self.input_adapters['semseg'].stride_level
            W *= self.input_adapters['semseg'].stride_level
        else:
            B, _, H, W = list(x.values())[0].shape  # TODO: Deal with case where not all have same shape

        # Encode selected inputs to tokens
        input_task_tokens  = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }

        input_info = self.generate_input_info(input_task_tokens, image_size=(H, W))
        input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)
        
        return input_tokens, input_info 

    def plot_index_distribution(self, seg_idx_list, depth_idx_list, title , filename):
        # Segmentation 태스크의 인덱스 분포
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(seg_idx_list.cpu().numpy().flatten(), bins=range(self.pool_size + 1), alpha=0.7, label='Seg')
        plt.title('Segmentation Task Prompt Index Distribution')
        plt.xlabel('Index')
        plt.ylabel('Frequency')
        
        # Depth 태스크의 인덱스 분포
        plt.subplot(1, 2, 2)
        plt.hist(depth_idx_list.cpu().numpy().flatten(), bins=range(self.pool_size + 1), alpha=0.7, label='Depth')
        plt.title('Depth Task Prompt Index Distribution')
        plt.xlabel('Index')
        plt.ylabel('Frequency')
        
        plt.suptitle(title)
        plt.legend()
        plt.savefig(filename)
        plt.close()
        
    def forward(self, x: Union[Dict[str, torch.Tensor], torch.Tensor], return_all_layers=False, **kwargs):
        """
        Forward pass through input adapters, transformer encoder and output adapters.

        :param x: Input tensor or dictionary of tensors
        :param return_all_layers: Set to True to return all transformer layers
        """
        
        input_tokens, input_info  = self.process_input(x)
        
        prompt_input_size = self.top_k * self.prompt_length 
        
        global_tokens = self.global_tokens.expand(input_tokens.size(0), -1, -1)
        expanded_prompts_1 = self.task_specific_prompts_1.expand(input_tokens.size(0), -1, -1)
        # expanded_prompts_2 = self.task_specific_prompts_2.expand(input_tokens.size(0), -1, -1)
        
        # original_prompts = torch.cat([expanded_prompts_1, expanded_prompts_2], dim=1)
        
        input_tokens = torch.cat([global_tokens,  input_tokens ], dim = 1)
        
        want_size = input_tokens.shape[1]
        
        # original_tokens = input_tokens
        
        if self.prompt_deep:
           
            # for i, layer in enumerate(self.encoder):
            #   if i==0 :
            #     prompt_output = self.prompt(input_tokens)
            #     input_tokens = prompt_output['prompted_embedding']
            #     input_tokens = self.prompt_dropout((input_tokens))
            #     input_tokens = layer(input_tokens)  
            #   else:
            #     input_tokens = input_tokens[:, prompt_input_size:, :] 
            #     prompt_output = self.prompt(input_tokens)
            #     input_tokens = prompt_output['prompted_embedding']
            #     input_tokens = self.prompt_dropout((input_tokens))
            #     input_tokens = layer(input_tokens)
                
            for i, layer in enumerate(self.encoder):
                if 0 <= i < 12  :
                    prompt_instance = self.layer_prompt_pools[i]

                    now_size = input_tokens.shape[1]
                    delete_size = now_size - want_size 
                    
                    input_tokens = input_tokens[:, delete_size:, :] 
                    
                    task1_tokens = torch.cat([expanded_prompts_1 ], dim = 1 )
                    # task2_tokens = torch.cat([expanded_prompts_2 ,  global_tokens], dim = 1 )

                    task1_prompt_pool = prompt_instance( task1_tokens)['prompted_embedding'][:,:prompt_input_size,:]
                    # task2_prompt_pool = prompt_instance( task2_tokens)['prompted_embedding'][:,:prompt_input_size,:]
                    
                    # depth_idx_list = prompt_instance( task1_tokens)['prompt_idx']
                    # seg_idx_list  = prompt_instance( task2_tokens)['prompt_idx']
                    
                    #Plot which idx selected 
                    #self.plot_index_distribution(seg_idx_list[i], depth_idx_list[i], title=f'Layer {i} Prompt Index Distribution', filename=f'layer_{i}_distribution.png')
                    
                    task1_prompt_pool = self.prompt_dropout(task1_prompt_pool)
                    
                    if i == 5 : 
                        input_tokens = torch.cat([ task1_prompt_pool , input_tokens ] , dim = 1)
                        
                        input_tokens = layer(input_tokens)     
                    else: 
                        input_tokens = torch.cat([task1_prompt_pool , input_tokens ] , dim = 1) #prompt pool *2 + input_tokens
                        
                        input_tokens = layer(input_tokens) 
                            
            encoder_tokens =  torch.cat([expanded_prompts_1 ,task1_prompt_pool , input_tokens] , dim = 1 )
            
        # Decode tokens for each task using task-specific output adapters
        preds = {
            domain: self.output_adapters[domain](
                encoder_tokens=encoder_tokens,
                input_info=input_info,
            )
            for domain in self.output_adapters
        }

        return preds


@register_model
def multivit_base(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        **kwargs):

    print(kwargs)

    model = MultiViT(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def multivit_large(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        **kwargs):
    model = MultiViT(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model