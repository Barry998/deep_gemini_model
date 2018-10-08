from model_lib import *
from data_pre_process import *
from ae_util import *


data_name = "0924"
class_num = 2
epoch = 200
train_percent = 0.7
batch_size = 64

train_loader, valid_loader, test_loader, data_configuration = generate_data_loader(data_name, train_percent, batch_size)

device = "cuda:9"
dynamic_features = data_configuration["ts_vars_amount"]
static_features = data_configuration["static_vars_amount"]
seq_length = data_configuration["seq_length"]


model_ae = AutoEncoder(dynamic_features, seq_length).to(device)
PATH = './check_points/ae_1.pt'
trained_parameters = torch.load(PATH)
model_ae.load_state_dict(trained_parameters)
model_ae.to(device)


def ae_predict(data_i):
    with torch.no_grad():
        dynamic_input, static_input, labels = data_i
        dynamic_input = dynamic_input.view(-1, 1200).to(device)
        dynamic_input = dynamic_input.to(device)
        static_input = static_input.to(device)
        _, predict_i = model_ae(dynamic_input, static_input)
    return predict_i


dynamic_hidden_dim = 128
lstm_layers = 3
learning_rate = 0.0001
dropout_prob = 0.1
sd_ratio = 1
cuda_name = device
cnn_kernel = 3

data_name = "0924"
class_num = 2
epoch = 3000
train_percent = 0.7
batch_size = 64

print("lstm_hidden_dim:", dynamic_hidden_dim, "lstm_layer:", lstm_layers, "learning_rate:", learning_rate,
      "sd_ratio:", sd_ratio, "cuda:", cuda_name, "drop_rate", dropout_prob)

device = torch.device(cuda_name)
# model = LstmTestBn(dynamic_features, dynamic_hidden_dim, lstm_layers, class_num, dropout_prob).to(device)
# model = ModelCombine(dynamic_features, static_features, dynamic_hidden_dim, lstm_layers, class_num, dropout_prob, sd_ratio).to(device)
# model = StaticOnly(static_features, dynamic_hidden_dim, class_num, dropout_prob).to(device)
# model = Cnn(dynamic_features, seq_length, class_num, dropout_prob, cnn_kernel).to(device)
# model = Cnn_lstm(dynamic_features, class_num, cnn_kernel).to(device)
model = LstmOnly(dynamic_features, dynamic_hidden_dim, lstm_layers, class_num, dropout_prob).to(device)
# model = Cnn_lstm_2(dynamic_features, class_num, cnn_kernel).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
print(model)


train_history = list()
valid_history = list()
test_history = list()
for batch_order in range(epoch):
    model.train()
    total_loss = 0
    total_acc = 0
    batches = 0
    for _, data_i in enumerate(train_loader, 0):
        if model.training:
            dynamic_ae = ae_predict(data_i)
            dynamic_input, static_input, labels = data_i
            dynamic_ae = dynamic_ae.view(-1, seq_length, dynamic_features)
            data_i = [dynamic_ae, static_input, labels]
            model, loss_i, acc_i = train_epoch(data_i, model, optimizer, criterion, device)
        else:
            dynamic_input, static_input, labels = data_i
            dynamic_ae = ae_predict(data_i)
            dynamic_ae = dynamic_ae.view(-1, seq_length, dynamic_features)
            data_i = [dynamic_ae, static_input, labels]
            model, loss_i, acc_i = valid_epoch(data_i, model, optimizer, criterion, device)
        total_loss += loss_i
        total_acc += acc_i
        batches += 1
    assert batches > 0
    total_loss_mean = total_loss / (batches + 1)
    total_acc_mean = total_acc / (batches + 1)

    loss_batch_i = total_loss_mean
    accuracy_batch_i = total_acc_mean
    print("%14s" % 'Train Epoch:', "%3d" % batch_order, "%10s" % 'Loss:', "%.4f" % loss_batch_i,
          "%8s" % "Acc:", "%6.3f%%" % float(accuracy_batch_i * 100), "%10s" % "Batches:", "%4d" % batches)
    train_history.append([loss_batch_i, accuracy_batch_i])

    model.eval()
    total_loss = 0
    total_acc = 0
    batches = 0
    for _, data_i in enumerate(valid_loader, 0):
        if model.training:
            dynamic_input, static_input, labels = data_i
            dynamic_ae = ae_predict(data_i)
            dynamic_ae = dynamic_ae.view(-1, seq_length, dynamic_features)
            data_i = [dynamic_ae, static_input, labels]
            model, loss_i, acc_i = train_epoch(data_i, model, optimizer, criterion, device)
        else:
            dynamic_input, static_input, labels = data_i
            dynamic_ae = ae_predict(data_i)
            dynamic_ae = dynamic_ae.view(-1, seq_length, dynamic_features)
            data_i = [dynamic_ae, static_input, labels]
            model, loss_i, acc_i = valid_epoch(data_i, model, optimizer, criterion, device)
        total_loss += loss_i
        total_acc += acc_i
        batches += 1
    assert batches > 0
    total_loss_mean = total_loss / (batches + 1)
    total_acc_mean = total_acc / (batches + 1)

    loss_batch_i = total_loss_mean
    accuracy_batch_i = total_acc_mean
    print("%14s" % 'Valid Epoch:', "%3d" % batch_order, "%10s" % 'Loss:', "%.4f" % loss_batch_i,
          "%8s" % "Acc:", "%6.3f%%" % float(accuracy_batch_i * 100), "%10s" % "Batches:", "%4d" % batches)
    valid_history.append([loss_batch_i, accuracy_batch_i])

    total_loss = 0
    total_acc = 0
    batches = 0
    for _, data_i in enumerate(test_loader, 0):
        if model.training:
            dynamic_input, static_input, labels = data_i
            dynamic_ae = ae_predict(data_i)
            dynamic_ae = dynamic_ae.view(-1, seq_length, dynamic_features)
            data_i = [dynamic_ae, static_input, labels]
            model, loss_i, acc_i = train_epoch(data_i, model, optimizer, criterion, device)
        else:
            dynamic_input, static_input, labels = data_i
            dynamic_ae = ae_predict(data_i)
            dynamic_ae = dynamic_ae.view(-1, seq_length, dynamic_features)
            data_i = [dynamic_ae, static_input, labels]
            model, loss_i, acc_i = valid_epoch(data_i, model, optimizer, criterion, device)
        total_loss += loss_i
        total_acc += acc_i
        batches += 1
    assert batches > 0
    total_loss_mean = total_loss / (batches + 1)
    total_acc_mean = total_acc / (batches + 1)

    loss_batch_i = total_loss_mean
    accuracy_batch_i = total_acc_mean
    print("%14s" % 'Test Epoch:', "%3d" % batch_order, "%10s" % 'Loss:', "%.4f" % loss_batch_i,
          "%8s" % "Acc:", "%6.3f%%" % float(accuracy_batch_i * 100), "%10s" % "Batches:", "%4d" % batches)
    test_history.append([loss_batch_i, accuracy_batch_i])
    print("-" * 80)










