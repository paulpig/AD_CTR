import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pandas as pd
import math

class VAE(nn.Module):

    def __init__(self, embed_size, device):
        super(VAE, self).__init__()

        Z_dim = X_dim = h_dim = embed_size
        self.Z_dim = Z_dim
        self.X_dim= X_dim
        self.h_dim = h_dim
        self.embed_size= embed_size
        self.device = device
        self.stdv = 1.0 / math.sqrt(embed_size)
        # stdv = 1.0 / self.emb_size
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform(m.weight)
                # nn.init.normal_(m.weight, std=1.e-4)
                nn.init.uniform_(m.weight, -self.stdv, self.stdv)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
        
        # =============================== Q(z|X) ======================================
        self.dense_xh = nn.Linear(X_dim, h_dim)
        init_weights(self.dense_xh)

        self.dense_hz_mu = nn.Linear(h_dim, Z_dim)
        init_weights(self.dense_hz_mu)

        self.dense_hz_var = nn.Linear(h_dim, Z_dim)
        init_weights(self.dense_hz_var)

        # =============================== P(X|z) ======================================
        self.dense_zh = nn.Linear(Z_dim, h_dim)
        init_weights(self.dense_zh)

        self.dense_hx = nn.Linear(h_dim, X_dim)
        init_weights(self.dense_hx)

    def Q(self, X):
        # h = nn.ReLU()(self.dense_xh(X))
        h = nn.LeakyReLU()(self.dense_xh(X))
        
        z_mu = self.dense_hz_mu(h)
        z_var = self.dense_hz_var(h)
        return z_mu, z_var

    def sample_z(self, mu, log_var):
        mb_size = mu.shape[0]
        # eps = Variable(torch.randn(mb_size, self.Z_dim)).to(self.device)
        eps = torch.randn_like(log_var)
        return mu + torch.exp(log_var / 2) * eps

    def P(self, z):
        # h = nn.ReLU()(self.dense_zh(z))
        h = nn.LeakyReLU()(self.dense_zh(z))
        X = self.dense_hx(h)
        return X