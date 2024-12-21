import jittor as jt 

from jittor import nn
from jittor.misc import _pair
import math
import logging
import numpy as np 
import models.configs as configs
import copy
from os.path import join as pjoin
from jittor.nn import CrossEntropyLoss
from scipy import ndimage

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def swish(x):
    return x*jt.sigmoid(x)
ACT2FN={"gelu":nn.gelu,"relu":nn.relu,"swish":swish}


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return jt.array(weights)


class Mlp(nn.Module):
    def __init__(self,config):
        super(Mlp,self).__init__()
        self.fc1=nn.Linear(config.hidden_size,config.transformer["mlp_dim"])
        self.act=ACT2FN["gelu"]
        self.fc2=nn.Linear(config.transformer["mlp_dim"],config.hidden_size)
        self.drop=nn.Dropout(config.transformer["dropout_rate"])
        
        # self._init_weights()

    # def _init_weights(self):
    #     jt.init.xavier_uniform_(self.fc1.weight)
    #     jt.init.xavier_uniform_(self.fc2.weight)
    #     jt.init.gauss_(self.fc1.bias, mean=0.0,std=1e-6)
    #     jt.init.gauss_(self.fc2.bias, mean=0.0,std=1e-6)
    
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
        self.num_attn_heads=config.transformer["num_heads"]
        self.attn_heads_size=int(config.hidden_size / self.num_attn_heads)
        self.scale=self.attn_heads_size ** -0.5
        self.all_head_size = self.num_attn_heads * self.attn_heads_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # self.qkv=nn.Linear(config.hidden_size,config.hidden_size * 3)
        self.attn_drop=nn.Dropout(config.transformer["attention_dropout_rate"])   
        self.out=nn.Linear(config.hidden_size,config.hidden_size)
        self.proj_drop=nn.Dropout(config.transformer["attention_dropout_rate"])    
        
    def execute(self,x):
        b,n,c=x.shape
        # qkv=self.qkv(x).reshape(b,n,3,self.num_attn_heads,self.attn_heads_size).permute(2,0,3,1,4)   #self.qkv->(b,n,3*c)   permute->(3,b,num_heads,n,head_size)
        # q,k,v=qkv[0],qkv[1],qkv[2]
        q=self.query(x).reshape(b,n,self.num_attn_heads,self.attn_heads_size).permute(0,2,1,3)
        k=self.key(x).reshape(b,n,self.num_attn_heads,self.attn_heads_size).permute(0,2,1,3)
        v=self.value(x).reshape(b,n,self.num_attn_heads,self.attn_heads_size).permute(0,2,1,3)
        attn=nn.bmm_transpose(q,k)   # ->(batch,num_heads,n,n)
        attn=nn.softmax(attn,dim=-1)
        weights=attn
        attn=self.attn_drop(attn)
        
        out=nn.bmm(attn,v)
        out=out.transpose(0,2,1,3).reshape(b,n,c)
        out=self.out(out)
        out=self.proj_drop(out)
        return out,weights
        
class EncoderBlock(nn.Module):
    def __init__(self,config):
        super(EncoderBlock,self).__init__()
        self.attn=Attention(config)
        self.dropout=nn.Dropout(config.transformer["dropout_rate"])
        self.attention_norm=nn.LayerNorm(config.hidden_size,eps=1e-5)
        self.ffn_norm=nn.LayerNorm(config.hidden_size,eps=1e-5)
        self.ffn=Mlp(config)
        self.hidden_size=config.hidden_size
    def execute(self,x):
        h=x
        x=self.attention_norm(x)
        x,weights=self.attn(x)
        x=x+h
        
        h=x
        x=self.ffn_norm(x)
        x=self.ffn(x)
        x=x+h
        return x,weights
    
    
    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with jt.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q , "kernel")]).reshape(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).reshape(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).reshape(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).reshape(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).reshape(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).reshape(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).reshape(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).reshape(-1)

            self.attn.query.weight.assign(query_weight)
            self.attn.key.weight.assign(key_weight)
            self.attn.value.weight.assign(value_weight)
            self.attn.out.weight.assign(out_weight)
            self.attn.query.bias.assign(query_bias)
            self.attn.key.bias.assign(key_bias)
            self.attn.value.bias.assign(value_bias)
            self.attn.out.bias.assign(out_bias)

            # mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            # mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            
            # mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            # mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()
            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")])
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")])

            self.ffn.fc1.weight.assign(mlp_weight_0)
            self.ffn.fc2.weight.assign(mlp_weight_1)
            self.ffn.fc1.bias.assign(mlp_bias_0)
            self.ffn.fc2.bias.assign(mlp_bias_1)

            self.attention_norm.weight.assign(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.assign(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.assign(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.assign(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

    
class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.transformer["num_layers"] - 1):
            layer = EncoderBlock(config)
            self.layer.append(copy.deepcopy(layer))
        self.part_select = Part_Attention()
        self.part_layer = EncoderBlock(config)
        self.part_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def execute(self, hidden_states):
        attn_weights = []
        for layer in self.layer:
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)            
        part_num, part_inx = self.part_select(attn_weights)
        part_inx = part_inx + 1
        parts = []
        B, num = part_inx.shape
        for i in range(B):
            parts.append(hidden_states[i, part_inx[i,:]])
        
        parts_stack=jt.stack(parts)
        
        if parts_stack.shape[1]==1:
            parts = parts_stack.squeeze(1)
        else:
            parts = parts_stack
        concat = jt.concat((hidden_states[:,0].unsqueeze(1), parts), dim=1)

        part_states, part_weights = self.part_layer(concat)
        part_encoded = self.part_norm(part_states)   

        return part_encoded

class Embeddings(nn.Module):
    def __init__(self,config,img_size,in_channels=3):
        super(Embeddings,self).__init__()
        img_size=_pair(img_size)
        patch_size=_pair(config.patches["size"])
        self.img_size=img_size
        self.patch_size=patch_size
        self.cls_token=jt.zeros((1,1,config.hidden_size))

        self.pos_drop=nn.Dropout(config.transformer["dropout_rate"])
        # non-overlap split
        if config.split == 'non-overlap':
            num_patches=int(img_size[1]/patch_size[1]) * int(img_size[0]/patch_size[0])
            self.patch_embeddings=nn.Conv(in_channels,config.hidden_size,kernel_size=patch_size,stride=patch_size)
            
        if config.split == 'overlap':
            num_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * ((img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = nn.Conv(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=(config.slide_step, config.slide_step))
        
        self.pos_embed=jt.zeros((1,num_patches+1,config.hidden_size))   
    def execute(self,x):

        B,C,H,W=x.shape
        assert H == self.img_size[0] and W == self.img_size[1],f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"

        
        x=self.patch_embeddings(x)
        
        x=x.flatten(2).transpose(0,2,1)
        
        _,i,j=self.cls_token.shape
        cls_tokens=self.cls_token.expand((B,i,j))
        x = jt.concat((cls_tokens, x), dim=1)

        embeddings = x + self.pos_embed
        embeddings = self.pos_drop(embeddings)
        return embeddings
    
    
    
    
class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def execute(self, x, target):
        logprobs = jt.nn.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def execute(self, input_ids):

        embedding_output = self.embeddings(input_ids)

        part_encoded = self.encoder(embedding_output)

        return part_encoded



class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, smoothing_value=0, zero_head=False):
        super(VisionTransformer, self).__init__()
        # self.patch_embeddings=Patch_Embedding(config,img_size=img_size)
        self.transformer=Transformer(config,img_size)
        # self.encoder=EncoderBlock(config)
        self.num_classes=num_classes
        self.smoothing_value=smoothing_value
        self.zero_head=zero_head
        self.classifier = config.classifier
        # num_patches=self.patch_embeddings.num_patches
        # self.cls_token=jt.zeros((1,1,config.hidden_size))
        # self.pos_embed=jt.zeros((1,num_patches+1,config.hidden_size))
        # self.pos_drop=nn.Dropout(config.transformer["dropout_rate"])
        
        self.part_head=nn.Linear(config.hidden_size,num_classes)
        # self.blocks=nn.ModuleList([
        #     EncoderBlock(config)
        #     for i in range(config.transformer["num_layers"]-1)
        # ])
    def execute(self,x,labels=None):
        # B=x.shape[0]
        # x=self.patch_embeddings(x)
        # _,i,j=self.cls_token.shape
        # cls_tokens=self.cls_token.expand((B,i,j))
        
        # x=jt.contrib.concat((cls_tokens,x),dim=1) 
        # x=x+self.pos_embed
        # x=self.pos_drop(x)
        
        # for block in self.blocks:
        #     x=block(x)
        
        # part_logits=self.part_head(x[:,0])

        part_tokens = self.transformer(x)

        part_logits = self.part_head(part_tokens[:, 0])
        
        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = LabelSmoothing(self.smoothing_value)
            part_loss = loss_fct(part_logits.view(-1, self.num_classes), labels.view(-1))
            
            contrast_loss = con_loss(part_tokens[:, 0], labels.view(-1))
            loss = part_loss + contrast_loss
            return loss, part_logits
        else:
            return part_logits
        
    def load_from(self, weights):
        with jt.no_grad():

            self.transformer.embeddings.patch_embeddings.weight.assign(np2th(weights["embedding/kernel"], conv=True))

            self.transformer.embeddings.patch_embeddings.bias.assign(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.assign(np2th(weights["cls"]))
            self.transformer.encoder.part_norm.weight.assign(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.part_norm.bias.assign(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.pos_embed
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.pos_embed.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.pos_embed.assign(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            # if self.transformer.embeddings.hybrid:          #    待补充
            #     self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
            #     gn_weight = np2th(weights["gn_root/scale"]).view(-1)
            #     gn_bias = np2th(weights["gn_root/bias"]).view(-1)
            #     self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
            #     self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                # for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                #     for uname, unit in block.named_children():
                #         unit.load_from(weights, n_block=bname, n_unit=uname) 
        
        
           
class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def execute(self, x):
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = jt.matmul(x[i], last_map)
        last_map = last_map[:,:,0,1:]
        
        #= last_map.max(2)
        max_inx,_=jt.argmax(last_map,2)
        return _, max_inx

        
def con_loss(features, labels):

    B, _= features.shape
    features = jt.misc.normalize(features)
    cos_matrix = features.matmul(features.t())
    pos_label_matrix = jt.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss        
        
        
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}
