# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Lower hls4ml model graphs to the AIE IR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from hls4ml.model.optimizer.optimizer import ModelOptimizerPass

from ...device_catalog import resolve_device
from ...ir import (
    BackendPolicies,
    LogicalIR,
    OpNode,
    TensorVar,
    ensure_backend_context,
    set_input_roles,
)
from ...ir.context import AIEBackendContext, ProjectConfig
from ...ir.graph import VIEW_FLATTEN_2D
from ...passes.utils import is_pointwise_dense
from .utils import _create_weight_tensors, _get_post_activation_precision, _precision_of, extract_layer_directives


class LowerToAieIr(ModelOptimizerPass):
    """Build the shared IR graph from the frontend model."""

    def __init__(self):
        self.name = 'lower_to_aie_ir'

    def transform(self, model) -> bool:
        ctx = ensure_backend_context(model, lambda: self._create_context(model))
        ctx.reset_ir()

        graph: LogicalIR = ctx.ir.logical
        layers = list(model.get_layers())
        input_var = model.get_input_variables()[0]
        batch_size = int(model.config.get_config_value('AIEConfig', {})['BatchSize'])
        batch_included = bool(ctx.policies.tensors_have_batch)

        def _canon(shape):
            dims = [int(x) for x in shape]
            return tuple(dims) if batch_included else tuple([batch_size] + dims)

        if input_var.name not in graph.tensors:
            graph.add_tensor(
                TensorVar(
                    name=input_var.name,
                    shape=_canon(input_var.shape),  # the batch axis every other tensor carries
                    precision=_precision_of(input_var),
                )
            )
        graph.mark_graph_input(input_var.name)

        for layer in layers:
            var = model.output_vars[layer.name]
            if var.name not in graph.tensors:
                prec = _precision_of(var)
                if layer.class_name == 'Dense' or is_pointwise_dense(layer):
                    # hls4ml may remove the following linear activation before lowering,
                    # leaving the Dense output var at accumulator precision.  Recover the
                    # correct post-activation precision from the config if possible.
                    override = _get_post_activation_precision(layer, model)
                    if override is not None:
                        prec = override
                graph.add_tensor(
                    TensorVar(
                        name=var.name,
                        shape=_canon(var.shape),
                        precision=prec,
                    )
                )

        node_map: Dict[str, OpNode] = {}
        param_tensors: Dict[str, tuple] = {}
        created_nodes = set()

        for layer in layers:
            if layer.class_name.lower() == 'input':
                continue
            node = OpNode(
                name=f'{layer.name}_aie',
                op_type=self._map_op_type(layer),
            )
            self._collect_metadata(layer, node)
            node.directives.update(extract_layer_directives(layer, model))

            if node.op_type in ('dense', 'conv2d'):
                weight_tv, bias_tv = _create_weight_tensors(layer, graph)
                if layer.class_name == 'DepthwiseConv2D':
                    # Keras keeps a depthwise filter per input channel, [kh, kw, Cin, multiplier];
                    # the canonical form is one group per channel, [kh, kw, Cin/groups, Cout].
                    data = np.asarray(weight_tv.data)
                    kh, kw, channels, multiplier = data.shape
                    weight_tv.data = data.reshape(kh, kw, 1, channels * multiplier)
                    weight_tv.shape = (kh, kw, 1, channels * multiplier)
                param_tensors[layer.name] = (weight_tv, bias_tv)

            if node.op_type == 'layer_norm':
                for weight_name, role in (('scale', 'gamma'), ('bias', 'beta')):
                    wvar = layer.weights.get(weight_name)
                    if wvar is None:
                        raise RuntimeError(f'{layer.name}: LayerNormalization missing {weight_name!r} weight.')
                    intent = _precision_of(wvar)
                    tv = TensorVar(
                        name=f'{layer.name}_{role}',
                        shape=np.asarray(wvar.data).shape,
                        precision=intent,
                        data=wvar.data,
                    )
                    graph.add_tensor(tv)
                    if weight_name == 'scale':
                        gamma_tv = tv
                    else:
                        beta_tv = tv
                param_tensors[layer.name] = (gamma_tv, beta_tv)

            var = model.output_vars[layer.name]
            tv = graph.tensors[var.name]
            tv.producer = node
            node.outputs.append(tv)
            if node.op_type == 'transpose':
                self._normalize_transpose_perm(node, tv.shape)
            graph.add_node(node)
            node_map[layer.name] = node
            created_nodes.add(layer.name)

        for layer in layers:
            if layer.name not in created_nodes:
                continue
            node = node_map[layer.name]

            for src in layer.inputs:
                var = input_var if src == 'input' else model.output_vars[src]
                tv = graph.tensors[var.name]
                node.inputs.append(tv)
                tv.consumers.append(node)

            if layer.name in param_tensors:
                weight_tv, bias_tv = param_tensors[layer.name]
                for param_tv in (weight_tv, bias_tv):
                    if param_tv is not None:
                        node.inputs.append(param_tv)
                        param_tv.consumers.append(node)

            role_names = node.metadata.pop('input_roles', None)
            if role_names:
                set_input_roles(node, node.inputs, role_names)

        for out_var in model.get_output_variables():
            graph.mark_graph_output(out_var.name)

        return True

    @staticmethod
    def _input_rank(layer) -> int:
        """Rank of this layer's input tensor, batch axis included."""
        return len(layer.get_input_variable().shape) + 1

    def _collect_metadata(self, layer, node) -> None:
        meta: Dict[str, Any] = {}

        if layer.class_name == 'Dense' or is_pointwise_dense(layer):
            if layer.class_name == 'Dense':
                n_in = layer.get_attr('n_in')
                n_out = layer.get_attr('n_out')
            else:
                n_in = layer.get_attr('n_chan')
                n_out = layer.get_attr('n_filt')
            if n_in is None or n_out is None:
                raise ValueError(f'{layer.name}: missing n_in/n_out for {layer.class_name}.')
            meta['n_in'] = int(n_in)
            meta['n_out'] = int(n_out)
            has_bias = layer.get_attr('bias_data') is not None
            meta['input_roles'] = ['lhs', 'rhs'] + (['bias'] if has_bias else [])

        if node.op_type == 'conv2d':
            # hls4ml is channels-last and stores its weights as [kh, kw, Cin/groups, Cout], which
            # is the canonical conv2d contract; only the attribute names differ.
            meta.update(
                kernel_shape=(int(layer.get_attr('filt_height')), int(layer.get_attr('filt_width'))),
                strides=(int(layer.get_attr('stride_height', 1)), int(layer.get_attr('stride_width', 1))),
                dilations=(int(layer.get_attr('dilation_height', 1)), int(layer.get_attr('dilation_width', 1))),
                pads=tuple(int(layer.get_attr(f'pad_{side}', 0)) for side in ('top', 'left', 'bottom', 'right')),
                groups=int(layer.get_attr('n_chan')) if layer.class_name == 'DepthwiseConv2D' else 1,
            )
            meta['input_roles'] = ['lhs', 'rhs'] + (['bias'] if layer.get_attr('bias_data') is not None else [])

        if node.op_type == 'pool2d':
            # The canonical pool2d the ONNX MaxPool lowers to; hls4ml gives Keras 'same' pads explicitly.
            meta.update(
                kind=str(layer.get_attr('pool_op')).lower(),
                kernel_shape=(int(layer.get_attr('pool_height')), int(layer.get_attr('pool_width'))),
                strides=(int(layer.get_attr('stride_height')), int(layer.get_attr('stride_width'))),
                dilations=(1, 1),
                pads=tuple(int(layer.get_attr(f'pad_{side}', 0)) for side in ('top', 'left', 'bottom', 'right')),
            )
            meta['input_roles'] = ['lhs']

        if node.op_type == 'reshape':
            # hls4ml tensors are already in canonical order, so the flattened row ravels the axes
            # exactly as they are stored.
            meta['input_roles'] = ['lhs']
            meta['view'] = VIEW_FLATTEN_2D
            meta['axis_order'] = tuple(range(self._input_rank(layer)))

        if layer.class_name == 'ApplyAlpha':
            scale = layer.get_attr('scale_data')
            if scale is not None:
                meta['scale'] = np.asarray(scale, dtype=np.float64).flatten().tolist()

        if layer.class_name == 'LayerNormalization':
            meta['input_roles'] = ['lhs', 'gamma', 'beta']
            epsilon = layer.get_attr('epsilon')
            if epsilon is not None:
                meta['epsilon'] = float(epsilon)

        if layer.class_name == 'Activation':
            meta['input_roles'] = ['lhs']
            act = (layer.get_attr('activation', '') or '').lower()
            if act:
                meta['activation'] = act
        if layer.class_name == 'Transpose':
            meta['input_roles'] = ['lhs']
            perm = layer.get_attr('perm')
            if perm is None:
                raise ValueError(f'{layer.name}: missing Transpose perm attribute.')
            meta['perm'] = [int(x) for x in perm]
            meta['data_format'] = layer.get_attr('data_format')

        meta['layer_class'] = layer.class_name
        if is_pointwise_dense(layer):
            meta['source_class'] = layer.class_name
            meta['layer_class'] = 'Dense'
        meta['source_layer'] = layer.name

        if meta:
            node.metadata.update(meta)

    def _create_context(self, model) -> AIEBackendContext:
        config = model.config
        aie_cfg = config.get_config_value('AIEConfig', {}) or {}
        part_name = aie_cfg.get('Part') or config.get_config_value('Part') or aie_cfg.get('Device') or 'unknown_part'

        device, _ = resolve_device(part_name, aie_cfg)
        policies = BackendPolicies(
            fusion=config.get_config_value('AIEFusionPolicy', {}) or {},
            decomposition=config.get_config_value('AIEDecompositionPolicy', {}) or {},
            pack=config.get_config_value('AIEPackPolicy', {}) or {},
            cache=config.get_config_value('AIECachePolicy', {}) or {},
            tensors_have_batch=bool(
                (config.get_config_value('AIEFrontendPolicy', {}) or {}).get('TensorsHaveBatch', False)
            ),
        )

        project_config = ProjectConfig(
            output_dir=Path(config.get_output_dir()),
            project_name=config.get_project_name(),
            stamp=config.get_config_value('Stamp'),
            custom_sources=dict(config.backend.get_custom_source()),
        )
        return AIEBackendContext(device=device, policies=policies, project_config=project_config, aie_config=aie_cfg)

    def _map_op_type(self, layer) -> str:
        if layer.class_name in ('Dense',) or is_pointwise_dense(layer):
            return 'dense'
        if layer.class_name == 'SeparableConv2D':
            raise NotImplementedError(
                f'{layer.name}: a separable convolution is a depthwise and a pointwise convolution; '
                'split it in the model so each lowers to its own conv2d.'
            )
        if layer.class_name in ('Conv2D', 'DepthwiseConv2D'):
            return 'conv2d'
        if layer.class_name == 'Pooling2D':
            return 'pool2d'
        if layer.class_name in ('Reshape', 'Flatten'):
            return 'reshape'
        if layer.class_name == 'Transpose':
            return 'transpose'
        if layer.class_name == 'LayerNormalization':
            return 'layer_norm'
        return layer.class_name.lower()

    def _normalize_transpose_perm(self, node: OpNode, output_shape) -> None:
        perm = node.metadata.get('perm')
        rank = len([int(x) for x in output_shape])
        if len(perm) == rank - 1:
            node.metadata['perm'] = [0] + [int(p) + 1 for p in perm]
