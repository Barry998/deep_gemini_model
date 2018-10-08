import torch.nn as nn
import torch.nn.functional as F
import torch


def train_epoch(data_i, model, optimizer, criterion, device):
    dynamic_input, static_input, labels = data_i
    dynamic_input = dynamic_input.to(device)
    static_input = static_input.to(device)

    labels = labels.to(device).long()
    optimizer.zero_grad()

    out = model(dynamic_input, static_input)
    batch_loss = criterion(out, labels)
    batch_loss.backward()
    optimizer.step()

    batch_loss_number = batch_loss.item()
    _, predict_i = out.max(dim=1)
    correct_i = predict_i.eq(labels)
    accuracy = correct_i.sum().item() / len(labels)

    return model, batch_loss_number, accuracy


def valid_epoch(data_i, model, optimizer, criterion, device):
    with torch.no_grad():
        dynamic_input, static_input, labels = data_i
        dynamic_input = dynamic_input.to(device)
        static_input = static_input.to(device)

        labels = labels.to(device).long()
        optimizer.zero_grad()

        out = model(dynamic_input, static_input)
        batch_loss = criterion(out, labels)
        batch_loss_number = batch_loss.item()
        _, predict_i = out.max(dim=1)
        correct_i = predict_i.eq(labels)
        accuracy = correct_i.sum().item() / len(labels)
    return model, batch_loss_number, accuracy


def run_epoch(model, data_loader, optimizer, criterion, device):
    total_loss = 0
    total_acc = 0
    batches = 0
    for _, data_i in enumerate(data_loader, 0):
        if model.training:
            model, loss_i, acc_i = train_epoch(data_i, model, optimizer, criterion, device)
        else:
            model, loss_i, acc_i = valid_epoch(data_i, model, optimizer, criterion, device)
        total_loss += loss_i
        total_acc += acc_i
        batches += 1
    assert batches > 0
    total_loss_mean = total_loss / (batches + 1)
    total_acc_mean = total_acc / (batches + 1)
    return total_loss_mean, total_acc_mean, batches


class LstmTest(nn.Module):

    def __init__(self, input_features, lstm_hidden_dim, lstm_layers, class_num, drop_prob):
        super(LstmTest, self).__init__()
        self.drop_prob = drop_prob
        self.lstm = nn.LSTM(input_features, lstm_hidden_dim, lstm_layers, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.fc2 = nn.Linear(lstm_hidden_dim, 16)
        self.fc3 = nn.Linear(16, class_num)

    def forward(self, dynamic, static):
        out, (h, c) = self.lstm(dynamic)

        # h, c
        # h = h.view((-1, lstm_hidden_dim))

        # Output
        out = out[:, -1, :]

        # Output before Sigmoid
        # out = out[:,-1,:]
        # out = F.logits(out)
        out = nn.Dropout(self.drop_prob)(out)
        out = self.fc1(out)
        if self.training:
            out = nn.Dropout(self.drop_prob)(out)
        out = F.relu(out)
        out = self.fc2(out)
        if self.training:
            out = nn.Dropout(self.drop_prob)(out)
        out = F.relu(out)
        out = self.fc3(out)
        if self.training:
            out = nn.Dropout(self.drop_prob)(out)
        return out


class LstmOnly(nn.Module):

    def __init__(self, input_features, lstm_hidden_dim, lstm_layers, class_num, drop_prob):
        super(LstmOnly, self).__init__()
        self.drop_prob = drop_prob
        self.lstm = nn.LSTM(input_features, lstm_hidden_dim, lstm_layers, batch_first=True)
        self.fc = nn.Linear(int(lstm_hidden_dim), class_num)

        self.bn_lstm = nn.BatchNorm1d(lstm_hidden_dim)
        self.bn_fc = nn.BatchNorm1d(2)

    def forward(self, dynamic, static):
        out, (h, c) = self.lstm(dynamic)

        # h, c
        # h = h.view((-1, lstm_hidden_dim))
        # Output
        out = out[:, -1, :]
        out = self.bn_lstm(out)

        # Output before Sigmoid
        # out = out[:,-1,:]
        # out = F.logits(out)

        out = self.fc(out)
        out = self.bn_fc(out)
        out = F.relu(out)

        return out


class LstmTestBn(nn.Module):

    def __init__(self, input_features, lstm_hidden_dim, lstm_layers, class_num, drop_prob):
        super(LstmTestBn, self).__init__()
        self.drop_prob = drop_prob
        self.lstm = nn.LSTM(input_features, lstm_hidden_dim, lstm_layers, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.fc2 = nn.Linear(lstm_hidden_dim, int(lstm_hidden_dim/2))
        self.fc3 = nn.Linear(int(lstm_hidden_dim/2), class_num)

        self.bn_lstm = nn.BatchNorm1d(lstm_hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(lstm_hidden_dim)
        self.bn_fc2 = nn.BatchNorm1d(int(lstm_hidden_dim/2))
        self.bn_fc3 = nn.BatchNorm1d(2)

    def forward(self, dynamic, static):
        out, (h, c) = self.lstm(dynamic)

        # h, c
        # h = h.view((-1, lstm_hidden_dim))

        # Output
        out = out[:, -1, :]
        out = self.bn_lstm(out)

        # Output before Sigmoid
        # out = out[:,-1,:]
        # out = F.logits(out)

        out = self.fc1(out)
        out = self.bn_fc1(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = self.bn_fc2(out)
        out = F.relu(out)

        out = self.fc3(out)
        out = self.bn_fc3(out)
        return out


class ModelCombine(nn.Module):

    def __init__(self, input_features_dynamic, input_features_static, lstm_hidden_dim, lstm_layers, class_num, drop_prob, sd_ratio):
        super(ModelCombine, self).__init__()
        self.drop_prob = drop_prob
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(input_features_dynamic, lstm_hidden_dim, lstm_layers, batch_first=True)
        static_hidden = int(lstm_hidden_dim * sd_ratio)
        cat_hidden_input_1 = static_hidden + lstm_hidden_dim
        cat_hidden_output_1 = int(cat_hidden_input_1/2)
        cat_hidden_output_2 = int(cat_hidden_output_1/2)

        self.fc_dynamic = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.fc_static = nn.Linear(input_features_static, static_hidden)
        self.fc_cat_1 = nn.Linear(cat_hidden_input_1, cat_hidden_output_1)
        self.fc_cat_2 = nn.Linear(cat_hidden_output_1, cat_hidden_output_2)
        self.fc_cat_3 = nn.Linear(cat_hidden_output_2, class_num)

    def forward(self, dynamic, static):
        lstm_out, (lstm_h, lstm_c) = self.lstm(dynamic)
        # dynamic_out = lstm_out[:, -1, :]
        dynamic_out = lstm_h.view((-1, self.lstm_hidden_dim))

        dynamic_out = self.fc_dynamic(dynamic_out)
        if self.training:
            dynamic_out = nn.Dropout(self.drop_prob)(dynamic_out)
        dynamic_out = F.relu(dynamic_out)

        static_out = self.fc_static(static)
        if self.training:
            static_out = nn.Dropout(self.drop_prob)(static_out)
        static_out = F.relu(static_out)

        concat_out = torch.cat([dynamic_out, static_out], dim=1)
        concat_out = self.fc_cat_1(concat_out)
        if self.training:
            concat_out = nn.Dropout(self.drop_prob)(concat_out)
        concat_out = F.relu(concat_out)

        concat_out = self.fc_cat_2(concat_out)
        if self.training:
            concat_out = nn.Dropout(self.drop_prob)(concat_out)
        concat_out = F.relu(concat_out)

        concat_out = self.fc_cat_3(concat_out)
        if self.training:
            concat_out = nn.Dropout(self.drop_prob)(concat_out)
        final_out = F.relu(concat_out)

        return final_out


class StaticOnly(nn.Module):

    def __init__(self, input_features, hidden_layers, class_num, drop_prob):
        super(StaticOnly, self).__init__()
        self.drop_prob = drop_prob
        self.fc1 = nn.Linear(input_features, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, int(hidden_layers/2))
        self.fc3 = nn.Linear(int(hidden_layers/2), class_num)

    def forward(self, dynamic, static):
        out = self.fc1(static)
        if self.training:
            out = nn.Dropout(self.drop_prob)(out)
        out = F.relu(out)
        out = self.fc2(out)
        if self.training:
            out = nn.Dropout(self.drop_prob)(out)
        out = F.relu(out)
        out = self.fc3(out)
        if self.training:
            out = nn.Dropout(self.drop_prob)(out)
        return out


class Cnn(nn.Module):

    def __init__(self, input_features, seq_length, class_num, drop_prob, cnn_kernel_size=3):
        super(Cnn, self).__init__()
        self.drop_prob = drop_prob
        dims_pool_1_out = int((seq_length - cnn_kernel_size + 1) / 2)
        dims_pool_2_out = int((dims_pool_1_out - cnn_kernel_size + 1) / 2)
        dims_fc_1_in = int(dims_pool_2_out*input_features/2)

        self.cnn_1 = nn.Conv1d(input_features, int(input_features/2), kernel_size=cnn_kernel_size)
        self.cnn_2 = nn.Conv1d(int(input_features/2), int(input_features/2), kernel_size=cnn_kernel_size)
        self.fc_1 = nn.Linear(dims_fc_1_in, int(dims_fc_1_in/2))
        self.fc_2 = nn.Linear(int(dims_fc_1_in/2), int(dims_fc_1_in/4))
        self.fc_3 = nn.Linear(int(dims_fc_1_in/4), class_num)

        self.bn_1 = nn.BatchNorm1d(int(input_features/2))
        self.bn_2 = nn.BatchNorm1d(int(input_features/2))
        self.bn_3 = nn.BatchNorm1d(int(dims_fc_1_in/2))
        self.bn_4 = nn.BatchNorm1d(int(dims_fc_1_in/4))
        self.bn_5 = nn.BatchNorm1d(class_num)

    def forward(self, dynamic, static):
        dynamic = dynamic.transpose(1, 2)
        out = self.cnn_1(dynamic)
        out = self.bn_1(out)
        if self.training:
            out = nn.Dropout(self.drop_prob)(out)
        out = F.relu(out)
        out = F.max_pool1d(out, 2)

        out = self.cnn_2(out)
        out = self.bn_2(out)
        if self.training:
            out = nn.Dropout(self.drop_prob)(out)
        out = F.relu(out)
        out = F.max_pool1d(out, 2)

        out = out.view(out.size()[0], -1)
        out = self.fc_1(out)
        out = self.bn_3(out)
        if self.training:
            out = nn.Dropout(self.drop_prob)(out)
        out = F.relu(out)

        out = self.fc_2(out)
        out = self.bn_4(out)
        if self.training:
            out = nn.Dropout(self.drop_prob)(out)
        out = F.relu(out)

        out = self.fc_3(out)
        out = self.bn_5(out)

        return out


class Cnn_lstm(nn.Module):

    def __init__(self, input_features, class_num, cnn_kernel_size=3):
        super(Cnn_lstm, self).__init__()
        dims_fc_1_in = input_features*8

        self.cnn_1 = nn.Conv1d(input_features, input_features*2, kernel_size=cnn_kernel_size)
        self.cnn_2 = nn.Conv1d(input_features*2, input_features*4, kernel_size=cnn_kernel_size)
        self.lstm = nn.LSTM(input_features*4, input_features*8, 1, batch_first=True)

        self.fc_1 = nn.Linear(dims_fc_1_in, int(dims_fc_1_in/2))
        self.fc_2 = nn.Linear(int(dims_fc_1_in/2), int(dims_fc_1_in/4))
        self.fc_3 = nn.Linear(int(dims_fc_1_in/4), class_num)

        self.bn_1 = nn.BatchNorm1d(input_features*2)
        self.bn_2 = nn.BatchNorm1d(input_features*4)
        self.bn_3 = nn.BatchNorm1d(dims_fc_1_in)
        self.bn_4 = nn.BatchNorm1d(int(dims_fc_1_in/2))
        self.bn_5 = nn.BatchNorm1d(int(dims_fc_1_in/4))
        self.bn_6 = nn.BatchNorm1d(class_num)

    def forward(self, dynamic, static):
        dynamic = dynamic.transpose(1, 2)
        out = self.cnn_1(dynamic)
        out = self.bn_1(out)
        out = F.relu(out)
        out = F.max_pool1d(out, 2)

        out = self.cnn_2(out)
        out = self.bn_2(out)
        out = F.relu(out)
        out = F.max_pool1d(out, 2)

        out = out.transpose(1, 2)
        out, (_, _) = self.lstm(out)
        out = out[:, -1, :]
        out = self.bn_3(out)
        out = F.relu(out)

        out = out.view(out.size()[0], -1)
        out = self.fc_1(out)
        out = self.bn_4(out)
        out = F.relu(out)

        out = self.fc_2(out)
        out = self.bn_5(out)
        out = F.relu(out)

        out = self.fc_3(out)
        out = self.bn_6(out)

        return out


class Cnn_lstm_2(nn.Module):

    def __init__(self, input_features, class_num, cnn_kernel_size=3):
        super(Cnn_lstm_2, self).__init__()
        self.cnn_1 = nn.Conv1d(input_features, 2, stride=3, kernel_size=5)
        self.lstm = nn.LSTM(2, 4, 1, batch_first=True)
        self.fc = nn.Linear(4, class_num)

        self.bn_1 = nn.BatchNorm1d(2)
        self.bn_3 = nn.BatchNorm1d(4)
        self.bn_4 = nn.BatchNorm1d(2)

    def forward(self, dynamic, static):
        dynamic = dynamic.transpose(1, 2)
        out = self.cnn_1(dynamic)
        out = self.bn_1(out)
        out = F.relu(out)

        out = out.transpose(1, 2)
        out, (_, _) = self.lstm(out)
        out = out[:, -1, :]
        out = self.bn_3(out)
        out = F.relu(out)

        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        out = self.bn_4(out)

        return out



