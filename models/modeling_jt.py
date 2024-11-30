import jittor as jt 

from jittor import nn
from jittor.misc import _pair
import math
import numpy as np 
import models.configs as configs

def swish(x):
    return x*jt.sigmoid(x)
ACT2FN={"gelu":nn.GELU,"relu":nn.RELU,"swish":swish}

class Mlp(nn.Module):
    def __init__(self,config):
        super(Mlp,self).__init__()
        self.fc1=nn.Linear(config.hidden_size,config.transformer["mlp_dim"])
        self.act=ACT2FN["gelu"]
        self.fc2=nn.Linear(config.transformer["mlp_dim"],config.hidden_size)
        self.drop=nn.Dropout(config.transformer["dropout_rate"])
        
    def execute(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.drop(x)
        return x
    

class Attention(nn.Module):
    def __init__(self,config):
        super(Attention,self).__init__()
  
  
        
class EncoderBlock(nn.Module):
    def __init__(self,config):
        super(EncoderBlock,self).__init__()
        self.attn=Attention(config)
        self.dropout=nn.Dropout(config.transformer["dropout_rate"])
        self.norm=nn.LayerNorm(config.hidden_size,eps=1e-5)
        self.ffn=Mlp(config)
        
    def execute(self,x):
        h=x
        x=self.norm(x)
        x=self.attn(x)
        x=x+h
        
        h=x
        x=self.norm(x)
        x=self.fn(x)
        x=x+h
        return x
    

class Patch_Embedding(nn.Module):
    def __init__(self,config):
        super(Patch_Embedding,self).__init__()
        
    
    
    
    
class LabelSmoothing(nn.Module):
    pass

class VisionTransformer(nn.Module):
    pass


class Part_Attention(nn.Module):
    pass

        
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}
