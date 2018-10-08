import json as js
import torch
import torch.utils.data as torch_data


def load_data(data_name, data_path):
    training_data_name = "%s_training.json" % data_name
    data_description_name = "%s_description.json" % data_name
    config_output_name = "%s_config.json" % data_name
    training_data_index_name = "%s_training_index.json" % data_name

    training_data_output_path = "%s/%s/%s" % (data_path, data_name, training_data_name)
    description_output_path = "%s/%s/%s" % (data_path, data_name, data_description_name)
    config_output_path = "%s/%s/%s" % (data_path, data_name, config_output_name)
    training_data_index_output_path = "%s/%s/%s" % (data_path, data_name, training_data_index_name)

    with open(training_data_output_path, 'r') as js_file:
        training_data = js.load(js_file)
    with open(description_output_path, 'r') as js_file:
        data_description = js.load(js_file)
    with open(config_output_path, 'r') as js_file:
        data_configuration = js.load(js_file)
    with open(training_data_index_output_path, 'r') as js_file:
        data_index = js.load(js_file)

    training_data = filter_nan(training_data)
    training_data = torch.tensor(training_data, dtype=torch.float32)
    return training_data, data_description, data_configuration, data_index


def type_is_nan(data_i):
    data_valid = True
    for j in range(len(data_i)):
        if data_i[j] is None:
            data_valid = False
    return data_valid


def filter_nan(data_input):
    data_without_nan = list()
    for data_i in data_input:
        if type_is_nan(data_i):
            data_without_nan.append(data_i)
    return data_without_nan


def data_pre_process(data_input, data_configuration):
    dynamic_dim = data_configuration["ts_vars_amount"] * data_configuration["seq_length"]
    static_dim = data_configuration["static_vars_amount"] + dynamic_dim

    train_input = data_input[:, :static_dim]
    input_normalized, col_mean, col_std = data_normalize(train_input)

    data_dynamic = input_normalized[:, :dynamic_dim]
    data_dynamic = data_dynamic.reshape([-1, data_configuration["ts_vars_amount"], data_configuration["seq_length"]])
    data_dynamic = data_dynamic.transpose(1, 2)
    data_static = input_normalized[:, dynamic_dim:static_dim]
    data_output = data_input[:, static_dim:]
    data_output = label_output_method_1(data_output, 0.3)

    print(data_dynamic.size(), data_static.size(), data_output.size())
    return data_dynamic, data_static, data_output, col_mean, col_std


def data_normalize(data_input):
    col_mean = data_input.mean(0)
    col_std = data_input.std(0)
    input_normalized = (data_input - col_mean) / col_std
    return input_normalized, col_mean, col_std


def label_output_method_1(data_input, percent):
    total_obs = data_input.size()[0]
    top_number = int(total_obs * percent)

    top_k = data_input.topk(top_number, dim=0, largest=True)
    data_output_positive = data_input.ge(top_k[0][-1, ])
    data_output_positive = torch.tensor(data_output_positive, dtype=torch.int8)

    bottom_k = data_input.topk(top_number, dim=0, largest=False)
    data_output_negative = data_input.le(bottom_k[0][-1, ])
    data_output_negative = torch.tensor(data_output_negative, dtype=torch.int8)

    label_combine = data_output_positive - data_output_negative
    label_combine = label_combine.sum(dim=1)

    label_all_positive = label_combine.eq(2)
    label_all_negative = label_combine.eq(-2)

    return label_all_positive + label_all_negative * 2


def data_loader_format(data_input, data_configuration, eliminate_neutral=False):
    data_dynamic, data_static, data_output, col_mean, col_std = data_pre_process(data_input, data_configuration)
    assert data_dynamic.size()[0] == data_static.size()[0] == data_output.size()[0]

    data_output_total = list()
    for obs_i in range(len(data_dynamic)):
        if eliminate_neutral:
            if data_output[obs_i] != 0:
                data_output_total.append([data_dynamic[obs_i], data_static[obs_i], data_output[obs_i] - 1])
        else:
            data_output_total.append([data_dynamic[obs_i], data_static[obs_i], data_output[obs_i]])
    print(len(data_output_total) / len(data_dynamic))
    return data_output_total, col_mean, col_std


def generate_data_loader(data_name, train_percent, batch_size):
    working_path = "/home/data"
    train_data_name = "%s_train" % data_name
    training_data, data_description, data_configuration, data_index = load_data(train_data_name, working_path)
    training_data_processed, col_mean, col_std = data_loader_format(training_data, data_configuration, eliminate_neutral=True)
    test_data_name = "%s_test" % data_name
    test_data, data_description_2, data_configuration_2, data_index_2 = load_data(test_data_name, working_path)
    test_data_processed, col_mean_2, col_std_2 = data_loader_format(test_data, data_configuration_2, eliminate_neutral=True)

    train_set = training_data_processed[:int(len(training_data_processed)*train_percent)]
    valid_set = training_data_processed[int(len(training_data_processed)*train_percent):]

    train_loader = torch_data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch_data.DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)
    test_loader = torch_data.DataLoader(dataset=test_data_processed, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, data_configuration


