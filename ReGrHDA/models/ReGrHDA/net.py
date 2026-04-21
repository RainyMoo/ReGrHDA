from utils.make_mask import make_mask
from ops.fc import FC, MLP
from ops.layer_norm import LayerNorm
from models.ReGrHDA.mila import MILA_ED
from models.ReGrHDA.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from models.ReGrHDA.clip_loss import Return_itc

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------
def softmax_with_temperature(x, T):
    x = x / T
    exp_x = torch.exp(x)
    softmax = exp_x / torch.sum(exp_x)
    return softmax

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)
        #att = softmax_with_temperature(att, T=0.5)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

def info_nce_loss(f_rt, f_gt, temperature=0.1):
    # Normalize
    f_rt = F.normalize(f_rt, dim=1)  # shape [B, D]
    f_gt = F.normalize(f_gt, dim=1)

    # 相似度矩阵：S[i][j] = f_rt[i] · f_gt[j]
    sim_matrix = torch.matmul(f_rt, f_gt.T)  # shape [B, B]
    sim_matrix = sim_matrix / temperature

    # InfoNCE: 每个 i 的正对是 i，目标是最大化对角 log-softmax
    labels = torch.arange(f_rt.size(0)).to(f_rt.device)  # [0, 1, ..., B-1]
    loss_i2t = F.cross_entropy(sim_matrix, labels)
    loss_t2i = F.cross_entropy(sim_matrix.T, labels)

    return (loss_i2t + loss_t2i) / 2

class Gate_Fusion(nn.Module):
    def __init__(self, __C):
        super(Gate_Fusion, self).__init__()
        self.__C = __C
        self.gate = nn.Linear(__C.FLAT_OUT_SIZE * 2, 1)
    
    def forward(self, lang_feat, img_feat):
        sum_feat = torch.cat((lang_feat, img_feat), dim=1)
        VT = F.sigmoid(self.gate(sum_feat))
        img_feat = torch.mul(img_feat, VT)
        lang_feat_fanshu = torch.norm(lang_feat, p=2)
        img_feat_fanshu = torch.norm(img_feat, p=2)
        gama = min(lang_feat_fanshu / img_feat_fanshu, 1)
        proj_feat = lang_feat + gama * img_feat
        
        return proj_feat

# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

            # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)
   
        self.backbone = MILA_ED(__C)
        
        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        self.attflat_grid = AttFlat(__C)
        
        if self.__C.ITC_LOSS == 'True':
            self.itc_loss = Return_itc(__C)
            
        self.gate_fusion1 = Gate_Fusion(__C)
        self.gate_fusion2 = Gate_Fusion(__C)
        # self.gate_fusion3 = Gate_Fusion(__C)
        
        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

        self.proj_norm1 = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj1 = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)
        
        self.proj_norm2 = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj2 = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, frcn_feat, grid_feat, bbox_feat, grid_bbox_feat, w_feat, h_feat, region_align, region_iou, ques_ix, ques_tensor):
        #print(grid_feat.size())

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)
                   
        frcn_feat, frcn_feat_mask, grid_feat, grid_feat_mask, rg_align, rg_iou = self.adapter(frcn_feat, grid_feat, bbox_feat, grid_bbox_feat,w_feat, h_feat, region_align, region_iou)       
    
        
        # Backbone Framework
        
        lang_feat, frcn_feat, grid_feat = self.backbone(
            lang_feat,
            frcn_feat,
            grid_feat,
            lang_feat_mask,
            frcn_feat_mask,
            grid_feat_mask
        ) 

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )
    
        grid_feat = self.attflat_grid(
            grid_feat,
            grid_feat_mask
        )
        
        frcn_feat = self.attflat_img(
            frcn_feat,
            frcn_feat_mask
        )
        
        # Classification layers

        proj_feat1 = self.gate_fusion1(lang_feat, frcn_feat)
        
        proj_feat2 = self.gate_fusion2(lang_feat, grid_feat)

        proj_feat_region = self.proj_norm(proj_feat1)
        proj_feat_region = self.proj(proj_feat_region)
        
        proj_feat_grid = self.proj_norm1(proj_feat2)
        proj_feat_grid = self.proj1(proj_feat_grid)
        
        proj_feat = self.proj_norm2(proj_feat1 + proj_feat2)
        proj_feat = self.proj2(proj_feat)
        
        
        
        
        # contrave loss
        #n = lang_feat.size(0)
        #t = 2
        #labels = np.arange(n)
        #logits = np.dot(lang_feat, img_feat.T) * np.exp(t)
        #loss_i = torch.nn.CrossEntropyLoss(logits, labels, axis=0)
        #loss_j = torch.nn.CrossEntropyLoss(logits, labels, axis=1)
        #ct_loss = (loss_i+loss_j)/2
        
        # proj_feat = self.proj_norm(lang_feat + frcn_feat)
        # proj_feat = self.proj(proj_feat)
        
        if self.__C.ITC_LOSS == 'True':
            itc_loss1 = self.itc_loss(lang_feat, frcn_feat)
            itc_loss2 = self.itc_loss(lang_feat, grid_feat)
            
            #return proj_feat, frcn_feat, grid_feat, lang_feat
            return proj_feat, frcn_feat, grid_feat, proj_feat_region, proj_feat_grid, itc_loss1, itc_loss2

        else:
            return proj_feat

