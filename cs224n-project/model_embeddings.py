#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
#from highway import Highway
#from resblock import ResBlock
from odeblock import ODEblock

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        self.embed_size = embed_size
        self.vocab = vocab

        pad_token_idx = vocab['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), embedding_dim=50, padding_idx=pad_token_idx)
        self.CNN_ = CNN(embed_size, char_embed=50)
        #self.Highway_ = Highway(embed_size)
        #self.ResBlock_ = ResBlock(embed_size)
        self.ODEblock_ = ODEblock(embed_size)
        self.dropout = nn.Dropout(p=0.3)

        ### END YOUR CODE

    def forward(self, input_):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        sentence_length = input_.shape[0]
        batch_size = input_.shape[1]
        x_emb = self.embeddings(input_)
        x_reshaped = x_emb.permute(0, 1, 3, 2) ## HERE MAY HAVE TO PERMUTE 1 and 0 [sent_len and batch_size] if TRAINING NOT WORK!!!
        x_reshaped = x_reshaped.contiguous().view(x_reshaped.shape[0] * x_reshaped.shape[1], x_reshaped.shape[2], x_reshaped.shape[3])
        x_convout = self.CNN_.forward(x_reshaped)
        #x_highway = self.Highway_.forward(x_convout)
        #x_highway = self.ResBlock_.forward(x_convout)
        x_highway = self.ODEblock_.forward(x_convout)
        ## check x_highway
        x_word_emb = self.dropout(x_highway)
        x_word_emb = x_word_emb.view(sentence_length, batch_size, x_word_emb.shape[-1])
        return x_word_emb

        ### END YOUR CODE

