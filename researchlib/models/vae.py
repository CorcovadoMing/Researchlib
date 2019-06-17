from torch import nn
import torch

class VAEModel(nn.Module):
    def __init__(self, encoder, decoder, hidden_vector_len=100, z_vector_len=32):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_vector_len = hidden_vector_len
        self.z_vector_len = z_vector_len
        self.mu_fc = nn.Linear(hidden_vector_len, z_vector_len)
        self.logvar_fc = nn.Linear(hidden_vector_len, z_vector_len)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample(self, mu, logvar):
        return self.decoder(self.reparameterize(mu, logvar))
    
    def forward(self, x):
        hidden = self.encoder(x)
        mu, logvar = self.mu_fc(hidden), self.logvar_fc(hidden)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar