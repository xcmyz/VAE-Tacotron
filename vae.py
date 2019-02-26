import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import hparams as hp


class ReferenceEncoder(nn.Module):
    """Reference Encoder"""

    ##############################
    # inputs: (batch, seq_length, 80)
    ##############################

    ##############################
    # outputs: (batch, ref_enc_gru_size=128)
    ##############################

    def __init__(self):
        super(ReferenceEncoder, self).__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hp.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(hp.num_mels, 3, 2, 1, K)
        # print(hp.ref_enc_filters[-1])
        # print(hp.embedding_size // 2)
        # print(hp.embedding_size * 2)
        self.gru = nn.GRU(
            input_size=hp.ref_enc_filters[-1] * out_channels, hidden_size=hp.embedding_size // 2, batch_first=True)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, hp.num_mels)
        ##############################
        # out: (batch, 1, seq_length, 80)
        ##############################

        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)
            ##############################
            # out: (batch, 128, seq_length // 2**K, 80 // 2**K)
            ##############################

        out = out.transpose(1, 2)
        ##############################
        # out: (batch, seq_length // 2**K, 128, 80 // 2**K)
        ##############################
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)
        ##############################
        # out: (batch, seq_length // 2**K, 128 * (80 // 2**K))
        ##############################

        _, out = self.gru(out)
        # print(np.shape(out))

        # print(out.squeeze(0))
        # print(np.shape(out.squeeze(0)))
        out = torch.tanh(out)
        # print(np.shape(out))
        ##############################
        # out: (batch, 128)
        ##############################
        return out.squeeze(0)


class VAE(nn.Module, ):
    """VAE"""

    ##############################
    # inputs: (batch, ref_enc_gru_size)
    ##############################

    ##############################
    # z: (batch, embedding_size)
    ##############################

    def __init__(self, ref_enc_gru_size=hp.embedding_size // 2, hidden_size=256):
        super(VAE, self).__init__()
        self.embedding = hp.z_dim
        self.FC_h = nn.Linear(ref_enc_gru_size, hidden_size)
        self.FC1 = nn.Linear(hidden_size, self.embedding)
        self.FC2 = nn.Linear(hidden_size, self.embedding)
        self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm1d(self.embedding)

        self.reference_encoder = ReferenceEncoder()

    def encoder(self, inputs):
        hidden = self.relu(self.FC_h(inputs))
        return self.FC1(hidden), self.FC2(hidden)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs):
        if self.training:
            intput_ = self.reference_encoder(torch.transpose(inputs, 1, 2))
            mu, log_var = self.encoder(intput_)
            z = self.reparameterize(mu, log_var)
            # print(np.shape(z))
            # z = self.bn(z)
            # print(z)
            # print(np.shape(z))
            # print(z)
            # print(mu)
            # print(log_var)
        else:
            mu = -1
            log_var = -1
            ref_mat = torch.zeros((hp.batch_size, hp.z_dim))
            z = torch.randn_like(ref_mat)

        return z, mu, log_var
