import torch
import math
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np


class ResBlock2D(nn.Module):
    def __init__(self, n_c, kernel=3, dilation=1, p_drop=0.15):
        super(ResBlock2D, self).__init__()
        padding = self._get_same_padding(kernel, dilation)

        layer_s = list()
        layer_s.append(nn.Conv2d(n_c, n_c, kernel, padding=padding, dilation=dilation, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        layer_s.append(nn.ELU(inplace=False))

        layer_s.append(nn.Dropout(p_drop))

        layer_s.append(nn.Conv2d(n_c, n_c, kernel, dilation=dilation, padding=padding, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        self.layer = nn.Sequential(*layer_s)
        self.final_activation = nn.ELU(inplace=False)

    def _get_same_padding(self, kernel, dilation):
        return (kernel + (kernel - 1) * (dilation - 1) - 1) // 2

    def forward(self, x):
        out = self.layer(x)
        return self.final_activation(x + out)


class ResidualNetwork(nn.Module):
    def __init__(self, n_block, n_feat_in, n_feat_out, dilation=[1, 2, 4, 8], p_drop=0.15):
        super(ResidualNetwork, self).__init__()

        layer_s = list()
        if n_feat_in != n_feat_out:
            layer_s.append(nn.Conv2d(n_feat_in, n_feat_out, 1, bias=False))
            layer_s.append(nn.InstanceNorm2d(n_feat_out, affine=True, eps=1e-6))
            layer_s.append(nn.ELU(inplace=False))

        for i_block in range(n_block):
            d = dilation[i_block % len(dilation)]
            res_block = ResBlock2D(n_feat_out, kernel=3, dilation=d, p_drop=p_drop)
            layer_s.append(res_block)

        self.layer = nn.Sequential(*layer_s)

    def forward(self, x):
        output = self.layer(x)
        return output


class FeedForwardLayer(nn.Module):
    def __init__(self, d_feat, d_ff, activation_dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.d_feat = d_feat
        self.d_ff = d_ff
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(activation_dropout, inplace=False)
        self.fc1 = nn.Linear(d_feat, d_ff)
        self.fc2 = nn.Linear(d_ff, d_feat)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x


class TriangleMultiplicativeModule(nn.Module):
    def __init__(self, dim, orign_dim, mix='ingoing'):
        super(TriangleMultiplicativeModule, self).__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, orign_dim)
        self.right_proj = nn.Linear(dim, orign_dim)

        self.left_gate = nn.Linear(dim, orign_dim)
        self.right_gate = nn.Linear(dim, orign_dim)
        self.out_gate = nn.Linear(dim, orign_dim)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(orign_dim)
        self.to_out = nn.Linear(orign_dim, dim)

    def forward(self, x):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = self.to_out(out)

        out = out * out_gate
        return out


class CoevolExtractor(nn.Module):
    def __init__(self, n_feat_proj, n_feat_out):
        super(CoevolExtractor, self).__init__()

        self.norm_2d = nn.LayerNorm(n_feat_proj * n_feat_proj)
        # project down to output dimension (pair feature dimension)
        self.proj_2 = nn.Linear(n_feat_proj ** 2, n_feat_out)

    def forward(self, x_down, x_down_w):
        B, N, L = x_down.shape[:3]

        pair = torch.einsum('abij,ablm->ailjm', x_down, x_down_w)  # outer-product & average pool
        pair = pair.reshape(B, L, L, -1)
        pair = self.norm_2d(pair)
        pair = self.proj_2(pair)  # project down to pair dimension
        return pair


class SequenceWeight(nn.Module):
    def __init__(self, d_model, heads, dropout=0.15):
        super(SequenceWeight, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, msa):
        B, N, L = msa.shape[:3]

        msa = msa.permute(0, 2, 1, 3)
        tar_seq = msa[:, :, 0].unsqueeze(2)

        q = self.to_query(tar_seq).view(B, L, 1, self.heads, self.d_k).permute(0, 1, 3, 2, 4).contiguous()
        k = self.to_key(msa).view(B, L, N, self.heads, self.d_k).permute(0, 1, 3, 4, 2).contiguous()

        q = q * self.scale
        attn = torch.matmul(q, k)
        attn = F.softmax(attn, dim=-1)
        return self.dropout(attn)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, heads, k_dim=None, v_dim=None, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        if k_dim == None:
            k_dim = d_model
        if v_dim == None:
            v_dim = d_model

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scaling = 1 / math.sqrt(self.d_k)

        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(k_dim, d_model)
        self.to_value = nn.Linear(v_dim, d_model)
        self.to_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, query, key, value, return_att=False):
        batch, L1 = query.shape[:2]
        batch, L2 = key.shape[:2]
        q = self.to_query(query).view(batch, L1, self.heads, self.d_k).permute(0, 2, 1, 3)
        k = self.to_key(key).view(batch, L2, self.heads, self.d_k).permute(0, 2, 1, 3)
        v = self.to_value(value).view(batch, L2, self.heads, self.d_k).permute(0, 2, 1, 3)
        #
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        #
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, L1, -1)
        #
        out = self.to_out(out)
        if return_att:
            attention = 0.5 * (attention + attention.permute(0, 1, 3, 2))
            return out, attention.permute(0, 2, 3, 1)
        return out


class AxialEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, p_drop=0.1):
        super(AxialEncoderLayer, self).__init__()

        # multihead attention
        self.attn_L = MultiheadAttention(d_model, heads, dropout=p_drop)
        self.attn_N = MultiheadAttention(d_model, heads, dropout=p_drop)

        # feedforward
        self.ff = FeedForwardLayer(d_model, d_ff, activation_dropout=p_drop)

        # normalization module
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop, inplace=False)
        self.dropout2 = nn.Dropout(p_drop, inplace=False)
        self.dropout3 = nn.Dropout(p_drop, inplace=False)

    def forward(self, src, return_att=False):
        # Input shape for multihead attention: (BATCH, L, L, EMB)
        B, N, L = src.shape[:3]

        # attention over L
        src2 = self.norm1(src)
        src2 = src2.reshape(B * N, L, -1)
        src2 = self.attn_L(src2, src2, src2)
        src2 = src2.reshape(B, N, L, -1)
        src = src + self.dropout1(src2)

        # attention over N
        src2 = self.norm2(src)
        src2 = src2.permute(0, 2, 1, 3).reshape(B * L, N, -1)
        src2 = self.attn_N(src2, src2, src2)  # attention over N
        src2 = src2.reshape(B, L, N, -1).permute(0, 2, 1, 3)
        src = src + self.dropout2(src2)

        # feed-forward
        src2 = self.norm3(src)  # pre-normalization
        src2 = self.ff(src2)
        src = src + self.dropout3(src2)
        return src


class AttentionBlock(nn.Module):
    def __init__(
            self,
            d_feat,
            p_drop,
            n_att_head,
            r_ff,
    ):
        super(AttentionBlock, self).__init__()

        self.dropout_module1 = nn.Dropout(p_drop, inplace=False)
        self.dropout_module2 = nn.Dropout(p_drop, inplace=False)

        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim=d_feat, orign_dim=d_feat, mix='outgoing')
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim=d_feat, orign_dim=d_feat, mix='ingoing')
        self.axial_attention = AxialEncoderLayer(d_model=d_feat, d_ff=d_feat * r_ff,
                                                 heads=n_att_head, p_drop=p_drop)

    def forward(self, x):
        x = self.dropout_module1(self.triangle_multiply_outgoing(x)) + x
        x = self.dropout_module2(self.triangle_multiply_ingoing(x)) + x
        x = self.axial_attention(x)
        return x


class Att(nn.Module):
    def __init__(self, n_layers, d_feat=64, n_att_head=8, p_drop=0.1, r_ff=4):
        super(Att, self).__init__()

        layer_s = list()
        for _ in range(n_layers):
            res_block = AttentionBlock(d_feat, p_drop, n_att_head, r_ff)
            layer_s.append(res_block)

        self.layer = nn.Sequential(*layer_s)

    def forward(self, x):
        output = self.layer(x)
        return output


class Model_final(nn.Module):
    def __init__(self, d_features=256 + 144 + 128,
                 d_msa=768, d_pair=128, n_feat_proj=32, p_drop=0.15,
                 n_attlayer=3, n_block=61,
                 onebody_size=70 + 3 + 1 + 1,
                 twobody_size=33,
                 num_restype=20,
                 num_channel=128,
                 use_bfactor=False,
                 ):
        super(Model_final, self).__init__()
        # ==============================feat===============
        self.use_bfactor = use_bfactor
        if self.use_bfactor:
            self.onebody_size = onebody_size
        else:
            self.onebody_size = onebody_size - 1
        self.num_restype = num_restype
        self.twobody_size = twobody_size
        self.num_channel = num_channel

        self.add_module("retype", torch.nn.Conv3d(self.num_restype, 20, 1, padding=0, bias=False))
        self.add_module("conv3d_1", torch.nn.Conv3d(20, 20, 3, padding=0, bias=True))
        self.add_module("conv3d_2", torch.nn.Conv3d(20, 30, 4, padding=0, bias=True))
        self.add_module("conv3d_3", torch.nn.Conv3d(30, 10, 4, padding=0, bias=True))
        self.add_module("pool3d_1", torch.nn.AvgPool3d(kernel_size=4, stride=4, padding=0))

        self.add_module("conv1d_1",
                        torch.nn.Conv1d(640 + self.onebody_size, self.num_channel // 2, 1, padding=0, bias=True))
        self.add_module("conv2d_1",
                        torch.nn.Conv2d(self.num_channel + self.twobody_size, self.num_channel, 1, padding=0,
                                        bias=True))
        self.add_module("inorm_1", torch.nn.InstanceNorm2d(self.num_channel, eps=1e-06, affine=True))

        # ==============================Msa_feat================================================
        self.norm = nn.LayerNorm(768)
        self.proj_1 = nn.Linear(d_msa, 256)
        self.norm_1 = nn.LayerNorm(256)
        self.proj_2 = nn.Linear(256, 64)
        self.norm_2 = nn.LayerNorm(64)
        self.proj_3 = nn.Linear(64, n_feat_proj)
        self.norm_3 = nn.LayerNorm(n_feat_proj)

        self.encoder = SequenceWeight(n_feat_proj, 1, dropout=p_drop)
        self.coevol = CoevolExtractor(n_feat_proj, d_pair)

        self.norm_new = nn.LayerNorm(d_pair)
        # ==============================Msa_feat================================================

        self.norm_47 = nn.LayerNorm(47)

        self.ResNet1 = ResidualNetwork(n_block=1, n_feat_in=d_features, n_feat_out=64, p_drop=p_drop)
        self.Attention = Att(n_layers=n_attlayer)

        self.resNet = ResidualNetwork(n_block=n_block, n_feat_in=64, n_feat_out=64, p_drop=p_drop)

        self.conv2d_37 = nn.Conv2d(64, 37, 1)
        self.conv2d_25_o = nn.Conv2d(64, 25, 1)
        self.conv2d_25_t = nn.Conv2d(64, 25, 1)
        self.conv2d_13 = nn.Conv2d(64, 13, 1)

        self.conv2d_15 = nn.Conv2d(64, 1, 1)

    def forward(self, idx, val, obt, tbt, msas):

        msa = msas["msa_feats"]
        attention = msas["row_att"]

        # =========================================process_feat=================
        nres = obt.shape[0]

        x = scatter_nd(idx, val, (nres, 24, 24, 24, self.num_restype))
        x = x.permute(0, 4, 1, 2, 3)

        out_retype = self._modules["retype"](x)
        out_conv3d_1 = F.elu(self._modules["conv3d_1"](out_retype))
        out_conv3d_2 = F.elu(self._modules["conv3d_2"](out_conv3d_1))
        out_conv3d_3 = F.elu(self._modules["conv3d_3"](out_conv3d_2))
        out_pool3d_1 = self._modules["pool3d_1"](out_conv3d_3)

        flattened_1 = torch.flatten(out_pool3d_1.permute(0, 2, 3, 4, 1), start_dim=1, end_dim=-1)
        out_cat_1 = torch.cat([flattened_1, obt], dim=1).unsqueeze(0).permute(0, 2, 1)
        out_conv1d_1 = F.elu(self._modules["conv1d_1"](out_cat_1))

        temp1 = tile(out_conv1d_1.unsqueeze(3), 3, nres)
        temp2 = tile(out_conv1d_1.unsqueeze(2), 2, nres)
        out_cat_2 = torch.cat([temp1, temp2, tbt], dim=1)
        out_conv2d_1 = self._modules["conv2d_1"](out_cat_2)
        out_inorm_1 = F.elu(self._modules["inorm_1"](out_conv2d_1))

        out_inorm_1 = out_inorm_1.permute(0, 3, 2, 1)
        # =========================================process msa========================================
        B, N, L, _ = msa.shape
        msa = self.norm(msa)
        msa = F.elu(self.norm_1(self.proj_1(msa)))
        msa = F.elu(self.norm_2(self.proj_2(msa)))
        msa_down = F.elu(self.norm_3(self.proj_3(msa)))

        # ---------------------------------------------------------------------
        w_seq = self.encoder(msa_down).reshape(B, L, 1, N).permute(0, 3, 1, 2)
        feat_1d = w_seq * msa_down

        # outer product
        pair = self.coevol(msa_down, feat_1d)
        pair = self.norm_new(pair)

        # aggregated along dimension N
        feat_1d = feat_1d.sum(1)

        # Get the query sequence in msa
        query = msa_down[:, 0]

        # additional 1D features
        feat_1d = torch.cat((feat_1d, query), dim=-1)
        left = feat_1d.unsqueeze(2).repeat(1, 1, L, 1)
        right = feat_1d.unsqueeze(1).repeat(1, L, 1, 1)

        # =========================================process msa========================================

        attention = 0.5 * (attention + attention.permute(0, 1, 3, 2))
        attention = attention.permute(0, 2, 3, 1).contiguous()

        inputs = torch.cat((out_inorm_1, pair, left, right, attention), -1)
        inputs = inputs.permute(0, 3, 1, 2).contiguous()

        # ============================================net=============================================
        x = self.ResNet1(inputs)
        x = x.permute(0, 2, 3, 1).contiguous()
        x_att = self.Attention(x)
        x_att = x_att.permute(0, 3, 1, 2).contiguous()
        out = self.resNet(x_att)
        #
        mask_logits = self.conv2d_15(out)
        mask_logits = mask_logits[0]

        mask_logits = (mask_logits + mask_logits.permute(0, 2, 1)) / 2
        mask_prediction = torch.sigmoid(mask_logits)[0]

        return mask_prediction, mask_logits


def scatter_nd(indices, updates, shape):

    device = indices.device

    size = np.prod(shape)
    out = torch.zeros(size).to(device)
    temp = np.array([int(np.prod(shape[i + 1:])) for i in range(len(shape))])
    flattened_indices = torch.mul(indices.long(), torch.as_tensor(temp, dtype=torch.long).to(device)).sum(dim=1)
    out = out.scatter_add(0, flattened_indices, updates)

    return out.view(shape)


def tile(a, dim, n_tile):
    # Get on the same device as indices
    device = a.device

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))

    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        device)
    return torch.index_select(a, dim, order_index)
