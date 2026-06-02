import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers import WhisperModel
import math
import sys
sys.path.append('..')
from networks import base_models
from networks.beats import BEATsWrapper


class AVENet(nn.Module):

    def __init__(self, args):
        super(AVENet, self).__init__()

        self.img_backbone_type = args.img_backbone_type
        self.aud_backbone_type = args.aud_backbone_type
        self.use_vision_blocks = args.use_vision_blocks
        self.use_audio_blocks = args.use_audio_blocks
        self.freeze_dino = args.freeze_dino
        self.freeze_whisper = args.freeze_whisper
        self.freeze_beats = args.freeze_beats
        self.trimap = args.tri_map
        self.epsilon = args.epsilon
        self.epsilon2 = args.epsilon2
        self.tau = 0.03
        self.Neg = args.Neg

        # Image backbone
        if self.img_backbone_type == 'resnet18':
            self.img_backbone = base_models.resnet18(modal='vision', pretrained=False)

        elif self.img_backbone_type == 'dino_vit':
            self.img_backbone = AutoModel.from_pretrained(args.dino_model_name)
            if self.freeze_dino:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.use_vision_blocks:
                encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
                self.vision_blocks = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.img_proj = nn.Conv2d(768, 512, kernel_size=1)

        # Audio backbone
        self.aud_proj = None
        if self.aud_backbone_type == 'resnet18':
            self.aud_backbone = base_models.resnet18(modal='audio')

        elif self.aud_backbone_type == 'whisper':
            whisper_model = WhisperModel.from_pretrained(args.whisper_model_name)
            self.aud_backbone = whisper_model.encoder
            if self.freeze_whisper:
                for param in self.aud_backbone.parameters():
                    param.requires_grad = False
            if self.use_audio_blocks:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=whisper_model.config.d_model,
                    nhead=whisper_model.config.encoder_attention_heads,
                    batch_first=True)
                self.audio_blocks = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.aud_proj = nn.Linear(whisper_model.config.d_model, 512)

        elif self.aud_backbone_type == 'beats':
            self.aud_backbone = BEATsWrapper(args.beats_checkpoint)
            if self.freeze_beats:
                for param in self.aud_backbone.parameters():
                    param.requires_grad = False
            if self.use_audio_blocks:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.aud_backbone.output_dim,
                    nhead=self.aud_backbone.num_attention_heads,
                    batch_first=True)
                self.audio_blocks = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.aud_proj = nn.Linear(self.aud_backbone.output_dim, 512)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # Initialize from scratch only the necessary modules
        modules_to_init = []
        if self.aud_backbone_type == 'resnet18':
            modules_to_init.append(self.aud_backbone)
        elif self.aud_proj is not None:
            if self.use_audio_blocks:
                modules_to_init.append(self.audio_blocks)
            modules_to_init.append(self.aud_proj)

        if self.img_backbone_type == 'dino_vit':
            modules_to_init.append(self.img_proj)
        else:
            modules_to_init.append(self.img_backbone)

        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    nn.init.constant_(m.bias, 0)

    def forward(self, image, aud, return_embeddings=False):

        B = image.shape[0]
        mask = (1 - 100 * torch.eye(B,B)).to(image.device)
        
        # Image features
        if self.img_backbone_type == 'dino_vit':
            # Extract patch embeddings
            outputs = self.img_backbone(pixel_values=image, interpolate_pos_encoding=True)
            last_hidden_states = outputs.last_hidden_state  # B x P x 768 (P is num patches)

            # Optional vision transformer blocks
            if self.use_vision_blocks:
                last_hidden_states = self.vision_blocks(last_hidden_states) # B x P x 768
    
            # Drop CLS and register tokens (1+4)
            patch_embeddings = last_hidden_states[:, 5:, :] # B x P x 768
            
            # Reshape into spatial grid
            num_patches = patch_embeddings.shape[1]
            H = W = int(math.sqrt(num_patches))
            patch_grid = patch_embeddings.reshape(B, H, W, 768)
            img_feat = patch_grid.permute(0, 3, 1, 2) # B x 768 x H x W
            
            # Project to match audio feature depth
            img_feat = self.img_proj(img_feat) # B x 512 x H x W
        else:
            img_feat = self.img_backbone(image) # B x 512 x 14 x 14
            
        img_feat = F.normalize(img_feat, dim=1)
        
        # Audio features
        if self.aud_backbone_type == 'whisper':
            aud_outputs = self.aud_backbone(input_features=aud)
            audio_tokens = aud_outputs.last_hidden_state # B x T x C
            if self.use_audio_blocks:
                audio_tokens = self.audio_blocks(audio_tokens) # B x T x C
            aud_feat = audio_tokens.mean(dim=1) # B x C
            aud_feat = self.aud_proj(aud_feat)
        elif self.aud_backbone_type == 'beats':
            audio_tokens, padding_mask = self.aud_backbone(aud) # B x T x C
            if self.use_audio_blocks:
                audio_tokens = self.audio_blocks(
                    audio_tokens, src_key_padding_mask=padding_mask) # B x T x C
            if padding_mask is not None:
                audio_tokens = audio_tokens.masked_fill(
                    padding_mask.unsqueeze(-1), 0)
                valid_tokens = (~padding_mask).sum(dim=1).clamp_min(1).unsqueeze(-1)
                aud_feat = audio_tokens.sum(dim=1) / valid_tokens # B x C
            else:
                aud_feat = audio_tokens.mean(dim=1) # B x C
            aud_feat = self.aud_proj(aud_feat)
        else:
            aud_feat = self.aud_backbone(aud) # B x C x H x W
            aud_feat = self.maxpool(aud_feat).view(B,-1) # B x C
        aud_feat = F.normalize(aud_feat, dim=1)
        
        # Calculate similarity maps
        S_cross = torch.einsum('ncqa,ckhw->nkqa', [img_feat, aud_feat.T.unsqueeze(2).unsqueeze(3)])          # Every image against every audio B x B x H x W
        S_diag = S_cross.diagonal(dim1=0, dim2=1).permute(2, 0, 1).unsqueeze(1)                              # Audio visual simularity map B x 1 x H x W
        S_cross_pooled = self.maxpool(S_cross).view(B,B) # BxB

        # Pseudo-masks
        mask_pos_cross =  torch.sigmoid((S_cross - self.epsilon) / self.tau) # Positive region mask for cross image audio pairs
        mask_pos = mask_pos_cross.diagonal(dim1=0, dim2=1).permute(2, 0, 1).unsqueeze(1) # Positive region mask for diagonal image audio pairs
        neg = 1 - mask_pos                                                   # Negative region mask

        # Trimap logic for the negative mask
        if self.trimap:
            mask_pos2 = torch.sigmoid((S_diag - self.epsilon2) / self.tau)
            neg = 1 - mask_pos2
        else:
            neg = 1 - mask_pos

        sim_pos = (mask_pos * S_diag).view(*S_diag.shape[:2],-1).sum(-1) / (mask_pos.view(*mask_pos.shape[:2],-1).sum(-1))                                   # Positive Bx1 
        sim_neg_easy = ((mask_pos_cross * S_cross).view(*S_cross.shape[:2],-1).sum(-1) / mask_pos_cross.view(*mask_pos_cross.shape[:2],-1).sum(-1) ) * mask  # Easy negative BxB
        sim_neg_hard = (neg * S_diag).view(*S_diag.shape[:2],-1).sum(-1) / neg.view(*neg.shape[:2],-1).sum(-1)                                               # Hard negative Bx1

        if self.Neg:
            logits = torch.cat((sim_pos,sim_neg_easy,sim_neg_hard),1)/0.07
        else:
            logits = torch.cat((sim_pos,sim_neg_easy),1)/0.07 # 0.07 is temperature

        if return_embeddings:
            img_emb = self.maxpool(img_feat).view(B, -1)
            img_emb = F.normalize(img_emb, dim=1)
            img_emb_positive_mask_mean = (
                img_feat * mask_pos).sum(dim=(2, 3))
            img_emb_positive_mask_mean = F.normalize(
                img_emb_positive_mask_mean, dim=1)
            embeddings = {
                'image_emb': img_emb,
                'image_emb_positive_mask_mean': img_emb_positive_mask_mean,
                'audio_emb': aud_feat,
            }
            return S_diag, logits, mask_pos, neg, S_cross_pooled, embeddings

        return S_diag, logits, mask_pos, neg, S_cross_pooled
