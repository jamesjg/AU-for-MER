from functools import partial
import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torchvision

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding = 1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride = 2, padding = 1)
    )

class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h = h, w = w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class HTNet(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        heads,
        num_hierarchies,
        block_repeats,
        mlp_mult = 4,
        channels = 3,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2 #
        fmap_size = image_size // patch_size #
        blocks = 2 ** (num_hierarchies - 1)#

        seq_len = (fmap_size // blocks) ** 2   # sequence length is held constant across heirarchy
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))


        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            x = aggregate(x)
        return self.mlp_head(x)

# This function is to confuse three models
class Fusionmodel(nn.Module):
  def __init__(self):
    #  extend from original
    super(Fusionmodel,self).__init__()
    self.fc1 = nn.Linear(15, 3)
    self.bn1 = nn.BatchNorm1d(3)
    self.d1 = nn.Dropout(p=0.5)
    self.fc_2 = nn.Linear(6, 3)
    self.relu = nn.ReLU()
    # forward layers is to use these layers above
  def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
    fuse_four_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), 0)
    fuse_out = self.fc1(fuse_four_features)
    fuse_out = self.relu(fuse_out)
    fuse_out = self.d1(fuse_out) # drop out
    fuse_whole_four_parts = torch.cat(
        (whole_feature,fuse_out), 0)
    fuse_whole_four_parts = self.relu(fuse_whole_four_parts)
    fuse_whole_four_parts = self.d1(fuse_whole_four_parts)
    out = self.fc_2(fuse_whole_four_parts)
    return out
  
class BiAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
        output = torch.bmm(attn_weights, v).mean(dim=1)  # 聚合
        return output



class HTNet_AU(HTNet):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        heads,
        num_hierarchies,
        block_repeats,
        mlp_mult = 4,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dim = 128,
        N_AU = 24,
        n_frames = 32
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            heads=heads,
            num_hierarchies=num_hierarchies,
            block_repeats=block_repeats,
            mlp_mult=mlp_mult,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout
        )

        self.au_project = nn.Linear(emb_dim, dim)
        self.in_project = nn.Linear(N_AU, dim)
        self.pos_embedding = nn.Parameter(torch.randn(n_frames+1, dim))
        self.au_index_embedding = nn.Embedding(N_AU, embedding_dim=emb_dim)
        self.state_project = nn.Linear(N_AU, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, N_AU)
        )

        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim, dropout=dropout, activation='gelu'),
            2)

        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim, dropout=dropout, activation='gelu'),
            2
        )
        self.multi_scale_pooling = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.ReLU()
        )

        
        self.attention_pooling = BiAttentionPooling(dim)
        self.au_to_middle_fusion = nn.Linear(dim, 512) 
        self.au_to_top_fusion = nn.Linear(512, 256) 


    def forward(self, au, img):
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)

        B, T, N_au = au.shape # [B, T, 24]
        
        au_indices = torch.arange(N_au, device=au.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 24]
        au_indices = au_indices.expand(B, T, -1)  # [B, T, 24]
        au_index_embeds = self.au_index_embedding(au_indices)  # [B, T, 24, d_model]

        au_embedding = (au.unsqueeze(-1) * au_index_embeds) # [B, T, 24, d_model]
        au_embedding = self.au_project(au_embedding)  # [B, T, 24, d_model]


        # temporal transformer
        pos = self.pos_embedding[:T].unsqueeze(0).unsqueeze(0) # [1, 1, T, d_model]
        au_embedding = au_embedding.transpose(1, 2).contiguous()  # [B, 24, T, d_model]
        au_embedding = au_embedding + pos
        au_embedding = au_embedding.view(B * N_au, T, -1) # [B*24, T, d_model]
        au_embedding = self.temporal_transformer(au_embedding.transpose(0, 1)).transpose(0, 1)  # [B*24, T, d_model]
        au_embedding = au_embedding.mean(dim=1)  # [B*24, d_model]
        au_embedding = au_embedding.view(B, N_au, -1)  # [B, 24, d_model]

        # spatial transformer
        au_embedding = self.spatial_transformer(au_embedding.transpose(0, 1)).transpose(0, 1)  # [B, 24, d_model]
        au_embedding = self.au_to_middle_fusion(au_embedding)  

        left_top_feature = au_embedding[:, [0,2,4,6], :].mean(dim=1)
        right_top_feature = au_embedding[:, [1,3,5,7], :].mean(dim=1) 
        left_bottom_feature = au_embedding[:, [11, 13, 15, 16, 19, 20, 21], :].mean(dim=1) 
        right_bottom_feature = au_embedding[:, [12, 14, 15, 16, 19, 20, 21], :].mean(dim=1)
        global_feature = self.au_to_top_fusion(au_embedding[:, [17,22,23], :]).mean(dim=1).view(B, -1, 1, 1)

        four_part_feature = torch.stack([left_top_feature, right_top_feature, left_bottom_feature, right_bottom_feature], dim=-1).view(B, -1, 2, 2)


        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            if level == 1:
                x = x + four_part_feature
            elif level == 2:
                x = x + global_feature
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            x = aggregate(x)    # [B, C, H, W]

        return self.mlp_head(x)
