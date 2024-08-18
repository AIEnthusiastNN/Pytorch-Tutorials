# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 08:09:44 2024

@author: Dr. Nudrat Nida

Build a NN model
"""
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class NeuralNetwork:
    def __init__(self,i_size,h_size,o_size):
        self.weight1=np.random.rand(i_size,h_size)
        self.weight2=np.random.rand(h_size,o_size)
        
    def forward(self,input):
        hidden=sigmoid(np.dot(input,self.weight1))
        output=sigmoid(np.dot(hidden,self.weight2))
        return output
    
nn=NeuralNetwork(2, 4,1)

input=np.array([0.5,0.7])
output=nn.forward(input)
print(output)
