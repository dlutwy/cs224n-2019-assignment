#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
### YOUR CODE HERE for part 1i

class CNN(nn.Module):
    '''
    map from x_reshape to x_conv_out
    set kernel num to be equal to e_word
    '''
    def __init__(self, in_embed_size, out_embed_size):
        super(CNN, self).__init__()
        self.cnn = nn.Conv1d(in_embed_size, out_embed_size, kernel_size = 5)
        pass
    def forward(self, x_reshape):
        '''
        @param x_reshape shape:(batch_size, embed_char_size, char_num)

        @return x_conv_out shape:
        '''
        x_conv = self.cnn(x_reshape) # (b, embed_char_size, char_num-k+1)
        x_conv_relu = nn.functional.relu(x_conv) #(b, embed_char_size, char_num-k+1)
        x_conv_out, _ = torch.max(x_conv_relu,-1) #(b, embed_char_size)
        return x_conv_out

### END YOUR CODE

if __name__ == "__main__":
    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cnn = CNN(3,4)
    x_reshape = torch.randn((2,3,5))
    out = cnn(x_reshape)
    assert (out.shape == (2,4)), "Shape does not match"
    print("Shape check complete")

