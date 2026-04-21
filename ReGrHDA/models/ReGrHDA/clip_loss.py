import torch
import torch.nn as nn
import numpy as np



def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class Return_itc(nn.Module):
    def __init__(self,__C):
        super(Return_itc,self).__init__()
        self.__C = __C
        self.vision_embed_dim = __C.FLAT_OUT_SIZE
        self.text_embed_dim = __C.FLAT_OUT_SIZE
        self.projection_dim = int(__C.FLAT_OUT_SIZE / 4)
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale =  nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self,image_feat, lang_feat):
            
        image_embeds = self.visual_projection(image_feat)
        text_embeds = self.text_projection(lang_feat)
            
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = clip_loss(logits_per_text)
        
        return loss