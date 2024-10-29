# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from typing import Any, List, Optional, Union

import numpy as np
import packaging
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .config import RosaConfig
import spops
from .splinear import SparseLinear, SparseLinearT
from .rosa_functions import RoSALinearFunction

if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.33.0"):
    from transformers.integrations import deepspeed_config
else:
    from transformers.deepspeed import deepspeed_config

class RosaLayer(BaseTunerLayer):
    # # All names of layers that may contain (trainable) adapter weights
    # adapter_layer_names = ("rosa_A", "rosa_B", "rosa_embedding_A", "rosa_embedding_B", "rosa_spa")
    # # All names of other parameters that may contain adapter-related parameters
    # other_param_names = ("r", "d", "lora_alpha", "scaling", "lora_dropout")

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("rosa_A", "rosa_B", "rosa_E", "rosa_embedding_A", "rosa_embedding_B", "rosa_spa")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "d", "lora_alpha", "scaling", "lora_dropout", "ranknum")
    
    def __init__(self, base_layer: nn.Module, impl: str, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.d = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.rosa_A = nn.ParameterDict({})
        self.rosa_B = nn.ParameterDict({})

        # 添加的代码
        self.rosa_E = nn.ParameterDict({})
        self.ranknum = nn.ParameterDict({})

        # For SpA
        self.rosa_spa = nn.ModuleDict({})
        self.impl = impl

        # For Embedding layer
        self.rosa_embedding_A = nn.ParameterDict({})
        self.rosa_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

        # this dummy tenosr is added to the input to make sure the backward pass is not skipped
        # notice that this is not a parameter, so will not be updated by optimizer
        self.rosa_dummy = torch.tensor([0.], dtype=torch.bfloat16, requires_grad=True)

    def _add_dummy(self, x: torch.Tensor):
        with torch.no_grad():
            # make sure the dummy tensor is zero and requires grad
            self.rosa_dummy.zero_()
            if self.rosa_dummy.device != x.device:
                self.rosa_dummy = self.rosa_dummy.to(x.device)
            self.rosa_dummy.requires_grad = True
        return x + self.rosa_dummy.to(x.dtype)

    def _get_weight_shape(self):
        if isinstance(self.get_base_layer(), torch.nn.Embedding):
            return (self.in_features, self.out_features)
        return (self.out_features, self.in_features)

    def update_layer(self, adapter_name, r, d, lora_alpha, lora_dropout, spa_store_transpose, rosa_dtype, init_lora_weights, use_rslora):
        # This code works for linear layers, override for other layer types
        if r < 0:
            raise ValueError(f"`r` should be a non-negative integer value but the value passed is {r}")

        if d < 0 or d > 1:
            raise ValueError(f"`d` should be a value between 0 and 1 but the value passed is {d}")
        

        self.r[adapter_name] = r
        self.d[adapter_name] = d

        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters

        if r == 0:
            self.scaling[adapter_name] = 1.
        elif use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        
        rosa_dtype = torch.bfloat16 if rosa_dtype == 'bf16' else (torch.float16 if rosa_dtype == 'fp16' else torch.float32)
        # if r > 0:
        #     self.rosa_A[adapter_name] = nn.Linear(self.in_features, r, bias=False, dtype=rosa_dtype)
        #     self.rosa_B[adapter_name] = nn.Linear(r, self.out_features, bias=False, dtype=rosa_dtype)
        # else:
        #     self.rosa_A[adapter_name] = nn.Identity()
        #     self.rosa_B[adapter_name] = nn.Identity()

        # 修改的代码
        # Right singular vectors
        self.rosa_A[adapter_name] = nn.Parameter(torch.randn(r, self.in_features, dtype=rosa_dtype))
        # Singular values
        self.rosa_E[adapter_name] = nn.Parameter(torch.randn(r, 1, dtype=rosa_dtype))
        # Left singular vectors
        self.rosa_B[adapter_name] = nn.Parameter(torch.randn(self.out_features, r, dtype=rosa_dtype))
        # The current rank
        self.ranknum[adapter_name] = nn.Parameter(torch.randn(1), requires_grad=False)
        self.ranknum[adapter_name].data.fill_(float(r))
        self.ranknum[adapter_name].requires_grad = False
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        
        device = None
        dtype = None
        weight_shape = self._get_weight_shape()
        
        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    dtype = weight.dtype
                    # self.to(weight.device, dtype=weight.dtype)
                # else:
                #     self.to(weight.device)
                device = weight.device
                break
        
        assert None not in [device, weight_shape], "weight or qweight should be available"

        if d > 0:
            self.rosa_spa[adapter_name] = SparseLinear(
                density=d,
                shape=weight_shape,
                store_transpose=spa_store_transpose if self.impl == 'spmm' else False, # 'sp_add' does not requires the transpositions
                dtype=rosa_dtype
            )
        else:
            self.rosa_spa[adapter_name] = nn.Identity()

        self.to(device)
        # if dtype is not None:
        #     self.to(dtype)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if adapter_name in self.rosa_A.keys():
            nn.init.zeros_(self.rosa_E[adapter_name])
            nn.init.normal_(self.rosa_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.rosa_B[adapter_name], mean=0.0, std=0.02)

        if init_lora_weights is False:
            return

        if self.r[adapter_name] <= 0:
            return

        # if adapter_name in self.rosa_A.keys():
        #     if init_lora_weights is True:
        #         # initialize A the same way as the default for nn.Linear and B to zero
        #         # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
        #         nn.init.kaiming_uniform_(self.rosa_A[adapter_name].weight, a=math.sqrt(5))
        #     elif init_lora_weights.lower() == "gaussian":
        #         nn.init.normal_(self.rosa_A[adapter_name].weight, std=1 / self.r[adapter_name])
        #     else:
        #         raise ValueError(f"Unknown initialization {init_lora_weights=}")
        #     nn.init.zeros_(self.rosa_B[adapter_name].weight)
        # if adapter_name in self.rosa_embedding_A.keys():
        #     # initialize a the same way as the default for nn.linear and b to zero
        #     nn.init.zeros_(self.rosa_embedding_A[adapter_name])
        #     nn.init.normal_(self.rosa_embedding_B[adapter_name])

    def loftq_init(self, adapter_name):
        if self.r[adapter_name] <= 0:
            assert False, "LoftQ is only supported for r > 0"
            return

        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, rosa_A, rosa_B = loftq_init(weight, **kwargs)
        if adapter_name in self.rosa_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            self.rosa_A[adapter_name].weight.data = rosa_A
            self.rosa_B[adapter_name].weight.data = rosa_B
        if adapter_name in self.rosa_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            # self.rosa_embedding_A[adapter_name].weight.data = rosa_A
            # self.rosa_embedding_B[adapter_name].weight.data = rosa_B
            self.rosa_embedding_A[adapter_name].data = rosa_A
            self.rosa_embedding_B[adapter_name].data = rosa_B
        self.get_base_layer().weight.data = qweight

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return

        if self.r[adapter] <= 0:
            return

        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.rosa_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.rosa_A.keys():
                continue

            if scale is None:
                if self.r[active_adapter] > 0:
                    self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def set_spa_mask(self, mask):
        assert len(self.active_adapters) <= 1, 'at most one RoSA adapter is supported for now'
        assert len(self.active_adapters) == 1, 'set_spa_mask was called but no active adapter found'
        adapter = self.active_adapters[0]

        assert adapter in self.rosa_spa, 'set_spa_mask was called for an adapter that does not exist'
        spa_module = self.rosa_spa[adapter]
        assert spa_module is not None, 'set_spa_mask was called while there is no spa_module'

        spa_module.set_mask(mask)

    def _spa_exists(self, adapter):
        if adapter not in self.d or self.d[adapter] <= 0:
            return False
        if not self.rosa_spa[adapter].exists():
            return False
        return True

    def _convert_spa_to_dense(self, adapter):
        assert self._spa_exists(adapter), 'spa does not exist, but _convert_spa_to_dense was called'
        return self.rosa_spa[adapter].to_dense()

    def find_weight(self) -> torch.Tensor:
        base_layer = self.get_base_layer()
        for weight_name in ("weight", "qweight"):
            weight = getattr(base_layer, weight_name, None)
            if weight is not None:
                return weight

    def set_lora_requires_grad(self, req_grad: bool):
        for active_adapter in self.active_adapters:
            for param_dict in [self.rosa_embedding_A, self.rosa_embedding_B]:
                if active_adapter not in param_dict:
                    continue
                param = param_dict[active_adapter]
                param.requires_grad = req_grad
            
            for module_dict in [self.rosa_A, self.rosa_B]:
                if active_adapter not in module_dict:
                    continue
                module = module_dict[active_adapter]
                if not hasattr(module, "weight"):
                    continue
                module.weight.requires_grad = req_grad

    def set_spa_requires_grad(self, req_grad: bool):
        for active_adapter in self.active_adapters:
            if active_adapter not in self.rosa_spa:
                continue
            module = self.rosa_spa[active_adapter]
            module.values.requires_grad = req_grad


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, RosaLayer):
    # RoSA implemented in a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        d: float = 0.0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        impl: str = 'auto',
        spa_store_transpose: bool = True,
        rosa_dtype: str = 'bf16',
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if impl == 'auto':
            impl = 'sp_add'
        RosaLayer.__init__(self, base_layer, impl, **kwargs)

        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, d, lora_alpha, lora_dropout, spa_store_transpose, rosa_dtype, init_lora_weights, use_rslora)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.rosa_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.rosa_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        if self.r[adapter] > 0:
            device = self.rosa_B[adapter].weight.device
            dtype = self.rosa_B[adapter].weight.dtype
        else:
            device = self.rosa_spa[adapter].values.device
            dtype = self.rosa_spa[adapter].values.dtype

        output_tensor = None
        if self.r[adapter] > 0:
            # In case users wants to merge the adapter weights that are in
            # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
            # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
            cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

            weight_A = self.rosa_A[adapter].weight
            weight_B = self.rosa_B[adapter].weight
            # 添加代码
            weight_E = self.rosa_E[adapter].weight

            if cast_to_fp32:
                weight_A = weight_A.float()
                weight_B = weight_B.float()
                # 添加代码
                weight_E = weight_E.float()

            # output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
            # 修改代码
            output_tensor = (
                transpose(self.rosa_B[adapter] @ (self.rosa_A[adapter] * self.rosa_E[adapter]), self.fan_in_fan_out)
                * self.scaling[adapter]
                / (self.ranknum[adapter] + 1e-5)
            )

            if cast_to_fp32:
                output_tensor = output_tensor.to(dtype=dtype)

                # cast back the weights
                self.rosa_A[adapter].weight.data = weight_A.to(dtype)
                self.rosa_B[adapter].weight.data = weight_B.to(dtype)
                # 添加代码
                self.rosa_E[adapter].weight.data = weight_E.to(dtype)

        if self._spa_exists(adapter):
            spa_dense = self._convert_spa_to_dense(adapter).to(dtype)
            if output_tensor is None:
                output_tensor = spa_dense
            else:
                output_tensor += spa_dense

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            assert len(self.active_adapters) == 1, 'rosa only supports precisely one adapter'
            active_adapter = self.active_adapters[0]
            assert active_adapter in self.rosa_A.keys()

            if self.r[active_adapter] == 0 and not self._spa_exists(active_adapter):
                # we are collecting gradients while lora deos not exist
                # adding a dummy to the input to enable gradient propagation
                x = self._add_dummy(x)

            if self.impl == 'spmm' or not self._spa_exists(active_adapter): # sp_add implementation is suboptimal when spa does not exist
                result = self.base_layer(x, *args, **kwargs)
                
                if self.r[active_adapter] > 0:
                    rosa_A = self.rosa_A[active_adapter]
                    rosa_B = self.rosa_B[active_adapter]

                    # 添加代码
                    rosa_E = self.rosa_E[active_adapter]

                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]

                    # 添加代码
                    ranknum = self.ranknum[active_adapter] + 1e-5

                    x = x.to(rosa_A.weight.dtype)
                    # result += rosa_B(rosa_A(dropout(x))) * scaling
                    # 修改代码
                    result += (dropout(x) @ (rosa_A * rosa_E).T @ rosa_B.T) * scaling / ranknum

                if self._spa_exists(active_adapter):
                    spa_module = self.rosa_spa[active_adapter]
                    # x = x.to(spa_module.values.dtype)
                    result += spa_module(x)
            else:
                assert self.impl == 'sp_add', f'unknown rosa implementation {self.impl}'
                dropout = self.lora_dropout[active_adapter]
                dropout_rate = dropout.p if isinstance(dropout, nn.Dropout) else 0
                scaling = self.scaling[active_adapter]
                # 添加代码
                ranknum = self.ranknum[active_adapter] + 1e-5
                # 修改代码 添加参数 ranknum rosa_E
                result = RoSALinearFunction.apply(
                    x,
                    self.get_base_layer(),
                    getattr(self.rosa_A[active_adapter], 'weight', None),
                    getattr(self.rosa_B[active_adapter], 'weight', None),
                    getattr(self.rosa_E[active_adapter], 'weight', None),
                    ranknum,
                    getattr(self.rosa_spa[active_adapter], 'values', None),
                    getattr(self.rosa_spa[active_adapter], 'row_offs', None),
                    getattr(self.rosa_spa[active_adapter], 'row_idx', None),
                    getattr(self.rosa_spa[active_adapter], 'col_idx', None),
                    scaling,
                    dropout_rate,
                    self.training
                )
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "rosa." + rep


class Embedding(nn.Module, RosaLayer):
    # RoSA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        d: float = 0.0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        impl: str = 'auto', # ignored. only spmm implementation is supported for Embedding.
        spa_store_transpose: bool = True,
        rosa_dtype: str = 'bf16',
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        impl = 'spmm'
        RosaLayer.__init__(self, base_layer, impl)

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, d, lora_alpha, lora_dropout, spa_store_transpose, rosa_dtype, init_lora_weights, use_rslora)

    def update_layer(self, adapter_name, r, d, lora_alpha, lora_dropout, spa_store_transpose, rosa_dtype, init_lora_weights, use_rslora):
        if r < 0:
            raise ValueError(f"`r` should be a non-negative integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        rosa_dtype = torch.bfloat16 if rosa_dtype == 'bf16' else (torch.float16 if rosa_dtype == 'fp16' else torch.float32)
        if r > 0:
            # Actual trainable parameters
            weight_A = torch.randn((r, self.in_features), dtype=rosa_dtype)
            weight_B = torch.randn((self.out_features, r), dtype=rosa_dtype)
            self.rosa_embedding_A[adapter_name] = nn.Parameter(weight_A)
            self.rosa_embedding_B[adapter_name] = nn.Parameter(weight_B)
        else:
            self.rosa_embedding_A[adapter_name] = None
            self.rosa_embedding_B[adapter_name] = None

        if r == 0:
            self.scaling[adapter_name] = 1.
        elif use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        base_layer = self.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        assert weight is not None, "The base layer does not have a weight attribute"

        weight_shape = self._get_weight_shape()

        if d > 0:
            self.rosa_spa[adapter_name] = SparseLinearT(
                density=d,
                shape=weight_shape,
                store_transpose=spa_store_transpose if self.impl == 'spmm' else False, # 'sp_add' does not require transpositions
                dtype=rosa_dtype
            )
        else:
            self.rosa_spa[adapter_name] = nn.Identity()

        # self.to(weight.device, dtype=weight.dtype)
        self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.rosa_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.copy()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.rosa_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        if self.r[adapter] > 0:
            device = self.rosa_embedding_B[adapter].device
            dtype = self.rosa_embedding_A[adapter].dtype
        else:
            device = self.rosa_spa[adapter].values.device
            dtype = self.rosa_spa[adapter].values.dtype
        
        if self.r[adapter] > 0:
            # In case users wants to merge the adapter weights that are in
            # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
            # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
            cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

            weight_A = self.rosa_embedding_A[adapter]
            weight_B = self.rosa_embedding_B[adapter]

            if cast_to_fp32:
                weight_A = weight_A.float()
                weight_B = weight_B.float()

            output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

            if cast_to_fp32:
                output_tensor = output_tensor.to(dtype=dtype)

                # cast back the weights
                self.rosa_embedding_A[adapter] = weight_A.to(dtype)
                self.rosa_embedding_B[adapter] = weight_B.to(dtype)

        if self._spa_exists(adapter):
            spa_dense = self._convert_spa_to_dense(adapter).to(dtype)
            if output_tensor is None:
                output_tensor = spa_dense
            else:
                output_tensor += spa_dense

        return output_tensor

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            assert self.impl == 'spmm', 'only spmm implementation is supported for Emebedding'
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.rosa_embedding_A:
                    continue

                if self.r[active_adapter] > 0:
                    embedding_A = self.rosa_embedding_A[active_adapter].T
                    embedding_B = self.rosa_embedding_B[active_adapter].T
                    scaling = self.scaling[active_adapter]
                    after_A = self._embed(x, embedding_A)
                    result += (after_A @ embedding_B) * scaling

                if self._spa_exists(active_adapter):
                    # note that if r = 0, then we cannot generate masks for spa
                    # since nothing requires gradient during mask generation and 
                    # the dummy tensor that works in Linear layer does not work here
                    assert self.r[active_adapter] > 0, 'embedding layer does not support SpA alone'

                    base_layer = self.get_base_layer()
                    assert base_layer.padding_idx is None and base_layer.max_norm is None and not base_layer.scale_grad_by_freq and not base_layer.sparse
                    x_onehot = F.one_hot(x, base_layer.num_embeddings)
                    result += self.rosa_spa[active_adapter](x_onehot)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "rosa." + rep
    

def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    rosa_config: RosaConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(rosa_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = rosa_config.fan_in_fan_out = False
        kwargs.update(rosa_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)

    return new_module


class RankAllocator:
    """
    The RankAllocator for AdaLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """

    def __init__(self, model, peft_config, adapter_name):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1

        self.reset_ipt()
        self._set_budget_scheduler(model)

    def set_total_step(self, total_step):
        self.peft_config.total_step = total_step

    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {} 
        self.exp_avg_unc = {}

    def _set_budget_scheduler(self, model):
        self.init_bgt = 0
        self.name_set = set()
        for n, p in model.named_parameters():
            if f"rosa_A.{self.adapter_name}" in n:
                self.init_bgt += p.size(0)
                self.name_set.add(n.replace("rosa_A", "%s"))
        self.name_set = sorted(self.name_set)
        # The total final rank budget
        self.target_bgt = self.peft_config.target_r * len(self.name_set)

    def budget_schedule(self, step: int):
        tinit = self.peft_config.tinit
        tfinal = self.peft_config.tfinal
        total_step = self.peft_config.total_step
        # Initial warmup
        if step <= tinit:
            budget = self.init_bgt
            mask_ind = False
        # Final fine-tuning
        elif step > total_step - tfinal:
            budget = self.target_bgt
            mask_ind = True
        else:
            # Budget decreasing with a cubic scheduler
            mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
            budget = int((self.init_bgt - self.target_bgt) * (mul_coeff**3) + self.target_bgt)
            mask_ind = True if step % self.peft_config.deltaT == 0 else False
        return budget, mask_ind

    def update_ipt(self, model):
        # Update the sensitivity and uncertainty for every weight
        for n, p in model.named_parameters():
            if "lora_" in n and self.adapter_name in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    if deepspeed_config() is not None:
                        import deepspeed

                        grad = deepspeed.utils.safe_get_full_grad(p)
                        self.ipt[n] = (p * grad).abs().detach()
                    else:
                        self.ipt[n] = (p * p.grad).abs().detach()
                    # Sensitivity smoothing
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                    # Uncertainty quantification
                    self.exp_avg_unc[n] = (
                        self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                    )

    def _element_score(self, n):
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_budget(self, model, budget):
        value_ipt = {}
        vector_ipt = {}
        triplet_ipt = {}
        # Get the importance score for A, E, B
        for n, p in model.named_parameters():
            if f"rosa_A.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace("rosa_A", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"rosa_B.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace("rosa_B", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"rosa_E.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                name_m = n.replace("rosa_E", "%s")
                value_ipt[name_m] = entry_ipt

        all_score = []
        # Calculate the score for each triplet
        for name_m in vector_ipt:
            ipt_E = value_ipt[name_m]
            ipt_AB = torch.cat(vector_ipt[name_m], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_m % "rosa_E"
            triplet_ipt[name_E] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))

        # Get the threshold by ranking ipt
        mask_threshold = torch.kthvalue(
            torch.cat(all_score),
            k=self.init_bgt - budget,
        )[0].item()

        rank_pattern = {}
        # Mask the unimportant triplets
        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"rosa_E.{self.adapter_name}" in n:
                    p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0)
                    rank_pattern[n] = (~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
        return rank_pattern

    def update_and_allocate(self, model, global_step, force_mask=False):
        # # Update the importance score and allocate the budget
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model)
        budget, mask_ind = self.budget_schedule(global_step)
        # Allocate the budget according to importance scores
        if mask_ind or force_mask:
            rank_pattern = self.mask_to_budget(model, budget)
        else:
            rank_pattern = None
        return budget, rank_pattern

    def mask_using_rank_pattern(self, model, rank_pattern):
        # Mask the unimportant triplets
        is_adapter_name_truncated = False
        if self.adapter_name not in next(iter(rank_pattern.keys())):
            is_adapter_name_truncated = True

        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"rosa_E.{self.adapter_name}" in n:
                    key = n if not is_adapter_name_truncated else n.replace(f".{self.adapter_name}", "")
                    mask = torch.Tensor(rank_pattern[key]).unsqueeze(-1).to(p.device)
                    p.masked_fill_(~mask.bool(), 0.0)
