from data_pre_process import *
from model_lib import *


dynamic_hidden_dim = 16
lstm_layers = 3
learning_rate = 0.0001
dropout_prob = 0.1
sd_ratio = 1
cuda_name = 'cuda:8'
cnn_kernel = 3

data_name = "0924"
class_num = 2
epoch = 3000
train_percent = 0.7
batch_size = 64

print("lstm_hidden_dim:", dynamic_hidden_dim, "lstm_layer:", lstm_layers, "learning_rate:", learning_rate,
      "sd_ratio:", sd_ratio, "cuda:", cuda_name, "drop_rate", dropout_prob)

train_loader, valid_loader, test_loader, data_configuration = generate_data_loader(data_name, train_percent, batch_size)
dynamic_features = data_configuration["ts_vars_amount"]
static_features = data_configuration["static_vars_amount"]
seq_length = data_configuration["seq_length"]

device = torch.device(cuda_name)
# model = LstmTestBn(dynamic_features, dynamic_hidden_dim, lstm_layers, class_num, dropout_prob).to(device)
# model = ModelCombine(dynamic_features, static_features, dynamic_hidden_dim, lstm_layers, class_num, dropout_prob, sd_ratio).to(device)
# model = StaticOnly(static_features, dynamic_hidden_dim, class_num, dropout_prob).to(device)
# model = Cnn(dynamic_features, seq_length, class_num, dropout_prob, cnn_kernel).to(device)
# model = Cnn_lstm(dynamic_features, class_num, cnn_kernel).to(device)
# model = LstmOnly(dynamic_features, dynamic_hidden_dim, lstm_layers, class_num, dropout_prob).to(device)
model = Cnn_lstm_2(dynamic_features, class_num, cnn_kernel).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
print(model)


train_history = list()
valid_history = list()
test_history = list()
for batch_order in range(epoch):
    model.train()
    loss_batch_i, accuracy_batch_i, batches = run_epoch(model, train_loader, optimizer, criterion, device)
    print("%14s" % 'Train Epoch:', "%3d" % batch_order, "%10s" % 'Loss:', "%.4f" % loss_batch_i,
          "%8s" % "Acc:", "%6.3f%%" % float(accuracy_batch_i * 100), "%10s" % "Batches:", "%4d" % batches)
    train_history.append([loss_batch_i, accuracy_batch_i])

    model.eval()
    loss_batch_i, accuracy_batch_i, batches = run_epoch(model, valid_loader, optimizer, criterion, device)
    print("%14s" % 'Valid Epoch:', "%3d" % batch_order, "%10s" % 'Loss:', "%.4f" % loss_batch_i,
          "%8s" % "Acc:", "%6.3f%%" % float(accuracy_batch_i * 100), "%10s" % "Batches:", "%4d" % batches)
    valid_history.append([loss_batch_i, accuracy_batch_i])

    loss_batch_i, accuracy_batch_i, batches = run_epoch(model, test_loader, optimizer, criterion, device)
    print("%14s" % 'Test Epoch:', "%3d" % batch_order, "%10s" % 'Loss:', "%.4f" % loss_batch_i,
          "%8s" % "Acc:", "%6.3f%%" % float(accuracy_batch_i * 100), "%10s" % "Batches:", "%4d" % batches)
    test_history.append([loss_batch_i, accuracy_batch_i])
    print("-" * 80)


PATH = './check_points/ae_1.pt'
torch.save(model.state_dict(), PATH)


