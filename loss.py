import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hparams


class VAE_TacotronLoss(nn.Module):
    """Loss"""

    def __init__(self, device):
        super(VAE_TacotronLoss, self).__init__()
        self.Device = device

    def compare(self, out, stan):
        if np.shape(stan)[2] >= np.shape(out)[2]:
            return stan[:, :, :np.shape(out)[2]]
        else:
            frame_arr = np.zeros([np.shape(out)[0], np.shape(out)[1], np.shape(out)[
                                 2]-np.shape(stan)[2]], dtype=np.float32)
            return torch.Tensor(np.concatenate((stan.cpu(), frame_arr), axis=2)).to(self.Device)

    def forward(self, mel_output, mels, linear_output, specs, mu, log_var):
        mel_loss = torch.abs(mel_output - self.compare(mel_output, mels))
        mel_loss = torch.mean(mel_loss)
        linear_loss = torch.abs(
            linear_output - self.compare(linear_output, specs))
        linear_loss = torch.mean(linear_loss)

        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_div = 10.0 * kl_div

        return mel_loss, linear_loss, kl_div
