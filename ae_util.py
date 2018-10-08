import torch
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, dynamic_features, seq_length):
        super(AutoEncoder, self).__init__()

        self.dynamic_features = dynamic_features
        self.seq_length = seq_length
        self.dynamic_dims = dynamic_features * seq_length

        self.encoder = nn.Sequential(
            nn.Linear(self.dynamic_dims, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.dynamic_dims),
        )

    def forward(self, dynamic, static):
        encoded = self.encoder(dynamic)
        decoded = self.decoder(encoded)
        return encoded, decoded


def save_ae_model(model_name, model):
    model_parameter_path = './check_points/%s.pt' % model_name
    torch.save(model.state_dict(), model_parameter_path)


def load_ae_model(model_name, data_configuration, device):
    dynamic_features = data_configuration["ts_vars_amount"]
    static_features = data_configuration["static_vars_amount"]
    seq_length = data_configuration["seq_length"]

    model_ae = AutoEncoder(dynamic_features, seq_length).to(device)
    model_parameter_path = './check_points/%s.pt' % model_name
    trained_parameters = torch.load(model_parameter_path)
    model_ae.load_state_dict(trained_parameters)
    model_ae.to(device)
    return model_ae

