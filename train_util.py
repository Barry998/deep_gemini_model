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


