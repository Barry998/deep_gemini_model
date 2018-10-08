from data_pre_process import *
from model_lib import *
from ae_util import *


learning_rate = 0.0001
cuda_name = 'cuda:8'
cnn_kernel = 3

data_name = "0924"
class_num = 2
epoch = 200
train_percent = 0.7
batch_size = 64


train_loader, valid_loader, test_loader, data_configuration = generate_data_loader(data_name, train_percent, batch_size)

dynamic_features = data_configuration["ts_vars_amount"]
static_features = data_configuration["static_vars_amount"]
seq_length = data_configuration["seq_length"]

device = torch.device(cuda_name)
model = AutoEncoder(dynamic_features, seq_length).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)


def train_epoch_ae(data_i, model, optimizer, criterion, device):
    dynamic_input, static_input, labels = data_i
    dynamic_input = dynamic_input.view(-1, 1200).to(device)
    optimizer.zero_grad()

    encoded_out, decoded_out = model(dynamic_input, static_input)
    batch_loss = criterion(decoded_out, dynamic_input)
    batch_loss.backward()
    optimizer.step()

    batch_loss_number = batch_loss.item()
    return model, batch_loss_number


def valid_epoch_ae(data_i, model, optimizer, criterion, device):
    with torch.no_grad():
        dynamic_input, static_input, labels = data_i
        dynamic_input = dynamic_input.view(-1, 1200).to(device)
        optimizer.zero_grad()

        encoded_out, decoded_out = model(dynamic_input, static_input)
        batch_loss = criterion(decoded_out, dynamic_input)
        batch_loss_number = batch_loss.item()
    return model, batch_loss_number


def run_epoch_ae(model, data_loader, optimizer, criterion, device):
    total_loss = 0
    batches = 0
    for _, data_i in enumerate(data_loader, 0):
        if model.training:
            model, loss_i = train_epoch(data_i, model, optimizer, criterion, device)
        else:
            model, loss_i = valid_epoch(data_i, model, optimizer, criterion, device)
        total_loss += loss_i
        batches += 1
    assert batches > 0
    total_loss_mean = total_loss / (batches + 1)
    return total_loss_mean, batches


print(model)

train_history = list()
valid_history = list()
test_history = list()
for epoch_order in range(epoch):

    model.train()
    total_loss = 0
    batches = 0
    for _, data_i in enumerate(train_loader, 0):
        dynamic_input, static_input, labels = data_i
        dynamic_input = dynamic_input.view(-1, 1200).to(device)
        optimizer.zero_grad()
        encoded_out, decoded_out = model(dynamic_input, static_input)

        batch_loss = criterion(decoded_out, dynamic_input)
        batch_loss.backward()
        optimizer.step()
        batch_loss_number = batch_loss.item()

        total_loss += batch_loss_number
        batches += 1
        total_loss_mean = total_loss / (batches + 1)

    print("%14s" % 'Train Epoch:', "%3d" % epoch_order, "%10s" % 'Loss:', "%.4f" % total_loss_mean,
          "%10s" % "Batches:", "%4d" % batches)
    train_history.append([total_loss_mean])

    model.eval()
    total_loss = 0
    batches = 0
    with torch.no_grad():
        for _, data_i in enumerate(valid_loader, 0):
            dynamic_input, static_input, labels = data_i
            dynamic_input = dynamic_input.view(-1, 1200).to(device)
            optimizer.zero_grad()
            encoded_out, decoded_out = model(dynamic_input, static_input)

            batch_loss = criterion(decoded_out, dynamic_input)
            batch_loss_number = batch_loss.item()

            total_loss += batch_loss_number
            batches += 1
            total_loss_mean = total_loss / (batches + 1)

    print("%14s" % 'Valid Epoch:', "%3d" % epoch_order, "%10s" % 'Loss:', "%.4f" % total_loss_mean,
          "%10s" % "Batches:", "%4d" % batches)
    valid_history.append([total_loss_mean])

    total_loss = 0
    batches = 0
    with torch.no_grad():
        for _, data_i in enumerate(test_loader, 0):
            dynamic_input, static_input, labels = data_i
            dynamic_input = dynamic_input.view(-1, 1200).to(device)
            optimizer.zero_grad()
            encoded_out, decoded_out = model(dynamic_input, static_input)

            batch_loss = criterion(decoded_out, dynamic_input)
            batch_loss_number = batch_loss.item()

            total_loss += batch_loss_number
            batches += 1
            total_loss_mean = total_loss / (batches + 1)

    print("%14s" % 'Test Epoch:', "%3d" % epoch_order, "%10s" % 'Loss:', "%.4f" % total_loss_mean,
          "%10s" % "Batches:", "%4d" % batches)
    valid_history.append([total_loss_mean])
    print("-" * 80)


PATH = './check_points/ae_1.pt'
torch.save(model.state_dict(), PATH)


