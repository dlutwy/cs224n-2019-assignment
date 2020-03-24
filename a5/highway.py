#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, embed_size, bias = True):
        # Two nn.Linear
        super(Highway, self).__init__()
        self.proj = nn.Linear(embed_size, embed_size, bias)
        self.gate = nn.Linear(embed_size, embed_size, bias)

    def forward(self, x):
        x_proj = (self.proj(x))
        x_proj = nn.functional.relu(x_proj)
        x_gate = torch.sigmoid(self.gate(x))
        x_highway = x_gate* x_proj + (1-x_gate)*x
        return x_highway

### END YOUR CODE 

if __name__ == "__main__":
    m = Highway(3)
    x = torch.randn((5,3))
    # x = torch.tensor(range(-5,5)).float()
    print(x.size())
    print(m(x).size())