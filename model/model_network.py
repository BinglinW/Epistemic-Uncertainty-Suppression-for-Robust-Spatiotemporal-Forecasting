import torch
import torch.nn as nn

from model.layer import GATEncoder, GMLP
from model.attention import TemporalEncoder, PositionalEncoding
from model.resnet import ResNet
from layers.network import Network

try:
    from model.diffusion_model import DiffSTG
except Exception:
    DiffSTG = None


class STDiff(nn.Module):
    def __init__(self, config, n_units, n_heads, num_v, dropout=0.2, alpha=0.2):
        super(STDiff, self).__init__()

        self.is_use_diffusion_model = False
        self.is_use_gat_encoder = True
        self.is_use_temporal_encoder = True
        self.is_use_spatio_encoder = True
        self.is_use_net_work = True

        self.hidden_dim = n_units[-1]
        self.num_v = num_v
        self.T_h = config.T_h
        self.in_F = config.F

        if self.is_use_gat_encoder:
            self.gat = GATEncoder(n_units, n_heads, num_v, dropout=dropout, alpha=alpha)
            self.gat_proj = nn.Linear(n_units[0], n_units[-1])
            self.gat_cat = nn.Linear(n_units[-1] * 2, n_units[-1])
        else:
            self.gat = nn.Linear(n_units[0], n_units[-1])

        self.embed_dim = self.hidden_dim * num_v
        if self.is_use_temporal_encoder:
            self.t = TemporalEncoder(
                d_model=self.embed_dim,
                out_dim=self.embed_dim,
                n_heads=8,
                atten_layer=6
            )
        else:
            self.t = nn.Linear(self.embed_dim, self.embed_dim)

        if self.is_use_spatio_encoder:
            self.s = ResNet(config.T_h, config.T_h)
        else:
            self.s = nn.Linear(self.embed_dim, self.embed_dim)

        self.fusion = nn.Linear(self.hidden_dim, self.hidden_dim)

        if self.is_use_net_work:
            self.net = Network(config.T_h, config.T_h, config.T_h, 3, "end")
        else:
            self.net = nn.Linear(self.embed_dim * 2, self.embed_dim)

        self.PE = PositionalEncoding(max_len=config.T_h, d_model=self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)

        if self.is_use_diffusion_model:
            if DiffSTG is None:
                raise ImportError("DiffSTG is not available but is_use_diffusion_model=True.")
            self.diffusion = DiffSTG(config)
            diffusion_out_dim = self.hidden_dim
        else:
            self.diffusion = nn.Linear(self.hidden_dim, self.hidden_dim)
            diffusion_out_dim = self.hidden_dim

        self.decoder = GMLP(diffusion_out_dim)
        self.final_proj = nn.Linear(diffusion_out_dim, self.in_F)
        self.tanh = nn.GELU()

    def forward(self, x, pos_w, pos_d):
        if self.is_use_gat_encoder:
            enc = self.gat(x)
            proj = self.gat_proj(x)
            enc = self.gat_cat(torch.cat((enc, proj), dim=-1))
        else:
            enc = self.gat(x)

        B, T, V, H = enc.shape
        trans_enc = enc.view(B, T, -1)

        trans_enc = self.layer_norm(trans_enc)
        trans_enc = self.PE(trans_enc)

        t_hidden = self.t(trans_enc)
        s_hidden = self.s(trans_enc)

        if self.is_use_net_work:
            st_hidden = self.net(s_hidden, t_hidden)
        else:
            st_hidden = self.net(torch.cat((s_hidden, t_hidden), dim=-1))

        st_hidden = st_hidden.view(B, T, V, -1)

        hidden = self.fusion(st_hidden)
        hidden = self.layer_norm2(hidden)

        if self.is_use_diffusion_model:
            x_masked = torch.cat((hidden, torch.zeros_like(hidden)), dim=1).to(hidden.device)
            x_masked = x_masked.transpose(1, 3)
            n_samples = 2
            x_masked = self.tanh(x_masked)
            diff_hidden = self.diffusion((x_masked, pos_w, pos_d), n_samples)
            diff_hidden = torch.nan_to_num(diff_hidden, nan=-1)
            diff_hidden = diff_hidden.transpose(2, 4)
            diff_hidden = diff_hidden[:, 0]
        else:
            diff_hidden = self.diffusion(hidden)

        diff_hidden = self.tanh(diff_hidden)
        diff_hidden = self.decoder(diff_hidden)
        diff_hidden = self.final_proj(diff_hidden)
        return diff_hidden
