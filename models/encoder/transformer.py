import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, d_model) to hold positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Compute positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape: [d_model // 2]
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even index
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd index
        
        # Add a batch dimension and register buffer
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)  # This makes sure the tensor is not a model parameter but is persistent
        
    def forward(self, x):
        # Add positional encodings to the input tensor
        x = x + self.pe[:, :x.size(1), :]  # The shape of x is [B, seq_len, d_model]
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model=2048, nhead=8, dim_feedforward=2048, dropout=0.1, max_len=5000, gamma=1.0):
        super(TransformerLayer, self).__init__()

        # Positional encoding
        self.positional_encoding_A = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.positional_encoding_B = PositionalEncoding(d_model=d_model, max_len=max_len)
        
        # Cross attention
        self.cross_attention_li = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        
        # Cross attention
        self.cross_attention_pi = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # LayerNorms for normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        self.gamma = gamma  # Scaling factor for the cross-attention output
        
    def forward(self, A, B, C):
        # A: [B, 140, 2048], B: [B, 20, 2048]
        # Add positional encoding
        A = self.positional_encoding_A(A)  # Positional encoding for tensor A
        B = self.positional_encoding_B(B)  # Positional encoding for tensor B
        C = self.positional_encoding_B(C)  # Positional encoding for tensor B
        
        # Transpose for multi-head attention compatibility: (seq_len, batch_size, embed_dim)
        A = A.transpose(0, 1)  # Shape: [140, B, 2048]
        B = B.transpose(0, 1)  # Shape: [20, B, 2048]
        C = C.transpose(0, 1)  # Shape: [20, B, 2048]
        

        # Cross attention: query = B, key = A, value = A
        cross_attn_output, _ = self.cross_attention_li(query=A, key=B, value=B)

        A = A + self.dropout(cross_attn_output)
        A = self.norm1(A)
        
        # Cross attention: query = B, key = A, value = A
        cross_attn_output, _ = self.cross_attention_pi(query=A, key=C, value=C)

        A = A + self.dropout(cross_attn_output) * self.gamma
        A = self.norm4(A)

        # Self-attention for tensor A
        A_sa, _ = self.self_attention(query=A, key=A, value=A)
        
        # Add & Norm for self-attention
        A_sa = A + self.dropout(A_sa)
        A_sa = self.norm2(A_sa)

        A_sa = A_sa.transpose(0, 1)  # Back to [B, 140, 2048]

        # FFN applied to A
        A_ffn = self.ffn(A_sa)

        # Add & Norm for FFN
        A_ffn = A_sa + self.dropout(A_ffn)
        A_out = self.norm3(A_ffn)

        # Final output
        return A_out, cross_attn_output
    

class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model, 
                 nhead=8, 
                 dim_feedforward=2048, 
                 dropout=0.1, 
                 max_len=5000):
        super(TransformerBlock, self).__init__()
        
        self.d_model = d_model  # list; [64, 256, 512, 1024, 2048]
        self.nhead = nhead
        self.dropout = dropout
        self.alpha = 0.5

        # llava feats proj layers
        llava_proj_layers = []
        for di in range(len(d_model)-1, 1, -1):
            proj_layer = nn.Linear(5120, d_model[di])
            llava_proj_layers.append(proj_layer)

        self.llava_proj_layers = nn.ModuleList(llava_proj_layers)
        
        # point feats proj layers
        point_proj_layers = []
        for di in range(len(d_model)-1, 1, -1):
            proj_layer = nn.Linear(1024, d_model[di])
            point_proj_layers.append(proj_layer)

        self.point_proj_layers = nn.ModuleList(point_proj_layers)

        trans_layers = []
        for di in range(len(d_model)-1, 1, -1):
            layer = TransformerLayer(d_model=d_model[di], 
                                     nhead=nhead, 
                                     dim_feedforward=d_model[di], 
                                     dropout=dropout, 
                                     max_len=max_len)
            trans_layers.append(layer)

        self.trans_layers = nn.ModuleList(trans_layers)

        # convolutional layers for upsampling, containing x2 upsampling
        upsample_convs = []
        for di in range(len(d_model)-1, 1, -1):
            conv_module = nn.Sequential(
                nn.Conv2d(d_model[di], d_model[di-1], 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
            upsample_convs.append(conv_module)

        self.upsample_convs = nn.ModuleList(upsample_convs)

        
    def forward(self, i_feats, l_feats, p_feats):
        '''
            i_feats: list of encoded features; [64, 256, 512, 1024, 2048], each with shape [B, C, H, W]
            l_feats: llava feats; [B, 39, 5120]
        '''
        cross_attn_outputs = []
        all_outputs = []
        i_feat_prev = None
        for di, i_feat in enumerate(reversed(i_feats)):
            b, c, h, w = i_feat.shape

            if i_feat_prev is not None:
                i_feat = self.alpha * i_feat + (1 - self.alpha) * i_feat_prev

            i_feat = i_feat.view(b, c, -1).permute(0, 2, 1)

            # project llava feats
            l_feat = self.llava_proj_layers[di](l_feats)
            
            # project point feats
            p_feat = self.point_proj_layers[di](p_feats)

            i_feat, cross_attn_output = self.trans_layers[di](i_feat, l_feat, p_feat)
            i_feat = i_feat.permute(0, 2, 1).view(b, c, h, w)

            cross_attn_outputs.append(cross_attn_output)
            all_outputs.append(i_feat)
            i_feat_prev = self.upsample_convs[di](i_feat)

            if di >= 2:
                i_feat_last = self.alpha * i_feats[di-1] + (1 - self.alpha) * i_feat_prev
                all_outputs.append(i_feat_last)
                break

        return all_outputs, cross_attn_outputs