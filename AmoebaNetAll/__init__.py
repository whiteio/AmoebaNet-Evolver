"""AmoebaNet-D for ImageNet"""
from collections import OrderedDict
from typing import TYPE_CHECKING, Iterator, List, Tuple, Union, cast

import torch
from torch import Tensor, nn

from AmoebaNetAll.genotype_d import (D_NORMAL_CONCAT, D_NORMAL_OPERATIONS, D_REDUCTION_CONCAT,
                                D_REDUCTION_OPERATIONS)

from AmoebaNetAll.operations import FactorizedReduce

__all__ = ['amoebanet']

if TYPE_CHECKING:
    NamedModules = OrderedDict[str, nn.Module]
else:
    NamedModules = OrderedDict


def relu_conv_bn(in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 ) -> nn.Module:
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class Classify(nn.Module):

    def __init__(self, channels_prev: int, num_classes: int):
        super().__init__()
        self.pool = nn.AvgPool2d(7)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(channels_prev, num_classes)

    def forward(self, states: Tuple[Tensor, Tensor]) -> Tensor:  # type: ignore
        x, _ = states
        x = self.pool(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


class Stem(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.relu = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(3, channels, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        x = input
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class Cell(nn.Module):
    def __init__(self,
                 channels_prev_prev: int,
                 channels_prev: int,
                 channels: int,
                 reduction: bool,
                 reduction_prev: bool,
                 normal_ops: List,
                 reduction_ops: List,
                 ) -> None:
        super().__init__()

        NORMAL_OPS = normal_ops
        REDUCTION_OPS = reduction_ops
        N_CONCAT = D_NORMAL_CONCAT
        R_CONCAT = D_REDUCTION_CONCAT


        self.reduce1 = relu_conv_bn(in_channels=channels_prev, out_channels=channels)

        self.reduce2: nn.Module = nn.Identity()
        if reduction_prev:
            self.reduce2 = FactorizedReduce(channels_prev_prev, channels)
        elif channels_prev_prev != channels:
            self.reduce2 = relu_conv_bn(in_channels=channels_prev_prev, out_channels=channels)

        if reduction:
            self.indices, op_classes = zip(*REDUCTION_OPS)
            self.concat = R_CONCAT
        else:
            self.indices, op_classes = zip(*NORMAL_OPS)
            self.concat = N_CONCAT

        self.operations = nn.ModuleList()

        for i, op_class in zip(self.indices, op_classes):
            if reduction and i < 2:
                stride = 2
            else:
                stride = 1

            op = op_class(channels, stride)
            self.operations.append(op)

    def extra_repr(self) -> str:
        return f'indices: {self.indices}'

    def forward(self,  # type: ignore
                input_or_states: Union[Tensor, Tuple[Tensor, Tensor]],
                ) -> Tuple[Tensor, Tensor]:
        if isinstance(input_or_states, tuple):
            s1, s2 = input_or_states
        else:
            s1 = s2 = input_or_states

        skip = s1

        s1 = self.reduce1(s1)
        s2 = self.reduce2(s2)

        _states = [s1, s2]

        operations = cast(nn.ModuleList, self.operations)
        indices = cast(List[int], self.indices)

        for i in range(0, len(operations), 2):
            h1 = _states[indices[i]]
            h2 = _states[indices[i+1]]

            op1 = operations[i]
            op2 = operations[i+1]

            h1 = op1(h1)
            h2 = op2(h2)

            s = h1 + h2
            _states.append(s)

        return torch.cat([_states[i] for i in self.concat], dim=1), skip


def amoebanet(num_classes: int = 10,
               num_layers: int = 4,
               num_filters: int = 512,
               normal_ops: List = [],
               reduction_ops: List = [],
               ) -> nn.Sequential:
    # Builds model for specified type (1:A, 2:B, 3:C, 4:D)
    layers: NamedModules = OrderedDict()

    assert num_layers % 3 == 0
    repeat_normal_cells = num_layers // 3

    channels = num_filters // 4
    channels_prev_prev = channels_prev = channels
    reduction_prev = False

    def make_cells(normal_ops, reduction_ops, reduction: bool, channels_scale: int, repeat: int) -> Iterator[Cell]:
        nonlocal channels_prev_prev
        nonlocal channels_prev
        nonlocal channels
        nonlocal reduction_prev

        channels *= channels_scale

        for i in range(repeat):
            cell = Cell(channels_prev_prev,
                        channels_prev,
                        channels,
                        reduction,
                        reduction_prev,
                        normal_ops,
                        reduction_ops)

            channels_prev_prev = channels_prev
            channels_prev = channels * len(cell.concat)
            reduction_prev = reduction

            yield cell

    def reduction_cell(normal_ops, reduction_ops) -> Cell:
        return next(make_cells(normal_ops, reduction_ops, reduction=True, channels_scale=2, repeat=1))

    def normal_cells(normal_ops, reduction_ops) -> Iterator[Tuple[int, Cell]]:
        return enumerate(make_cells(normal_ops, reduction_ops, reduction=False, channels_scale=1, repeat=repeat_normal_cells))

    # Stem for ImageNet
    layers['stem1'] = Stem(channels)
    layers['stem2'] = reduction_cell(normal_ops=normal_ops, reduction_ops=reduction_ops)
    layers['stem3'] = reduction_cell(normal_ops=normal_ops, reduction_ops=reduction_ops)
    # AmoebaNet cells
    layers.update((f'cell1_normal{i+1}', cell) for i, cell in normal_cells(normal_ops=normal_ops, reduction_ops= reduction_ops))
    layers['cell2_reduction'] = reduction_cell(normal_ops=normal_ops, reduction_ops= reduction_ops)
    layers.update((f'cell3_normal{i+1}', cell) for i, cell in normal_cells(normal_ops=normal_ops, reduction_ops= reduction_ops))
    layers['cell4_reduction'] = reduction_cell(normal_ops=normal_ops, reduction_ops= reduction_ops)
    layers.update((f'cell5_normal{i+1}', cell) for i, cell in normal_cells(normal_ops=normal_ops, reduction_ops= reduction_ops))

    # Finally, classifier
    layers['classify'] = Classify(channels_prev, num_classes)

    return nn.Sequential(layers)

    
