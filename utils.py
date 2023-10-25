import concurrent.futures
import csv
import os
from sklearn.metrics import accuracy_score, log_loss, hinge_loss, confusion_matrix
import pandas as pd
import torch


def aggregate_weights(weights_list, all_n_samples):
    total_samples = sum(all_n_samples[client_id][0] for client_id in all_n_samples)

    aggregated_state_dict = {}
    for key in weights_list[0].keys():
        weighted_weights = [client_weights[key] * (all_n_samples[client_id][0] / total_samples)
                            for client_weights, client_id in zip(weights_list, all_n_samples.keys())]

        aggregated_state_dict[key] = sum(weighted_weights)

    return aggregated_state_dict


def calculate_weighted_global_measures(clients,
                                       all_n_samples,
                                       all_train_measures,
                                       all_val_measures,
                                       display_result=False):
    total_samples_train = sum(all_n_samples[id][0] for id in all_n_samples)
    total_samples_val = sum(all_n_samples[id][1] for id in all_n_samples)
    global_train_measures = {}
    global_val_measures = {}
    for key in all_train_measures:
        global_train_measures[key] = sum([all_train_measures[key][c.id] * all_n_samples[c.id][0]
                                          for c in clients]) / total_samples_train
        global_val_measures[key] = sum(
            [all_val_measures[key][c.id] * all_n_samples[c.id][1] for c in clients]) / total_samples_val

    if display_result:
        print("Global Train Set Loss:", global_train_measures['loss'])
        print("Global Train Set Accuracy:", global_train_measures['accuracy'])
        print("Global Train Set FPR:", global_train_measures['fpr'])
        print("Global Train Set TPR:", global_train_measures['tpr'])
        print("Global Train Set BER:", global_train_measures['ber'])
        print("Global Validation Set Loss:", global_val_measures['loss'])
        print("Global Validation Set Accuracy:", global_val_measures['accuracy'])
        print("Global Validation Set FPR:", global_val_measures['fpr'])
        print("Global Validation Set TPR:", global_val_measures['tpr'])
        print("Global Validation Set BER:", global_val_measures['ber'])

    return global_train_measures, global_val_measures


def calculate_global_measures(clients, all_train_measures, all_val_measures, display_result=False):
    num_clients = len(clients)
    global_train_measures = {}
    global_val_measures = {}

    for key in all_train_measures:
        global_train_measures[key] = sum(all_train_measures[key][c.id] for c in clients) / num_clients
        global_val_measures[key] = sum(all_val_measures[key][c.id] for c in clients) / num_clients

    if display_result:
        print("Global Train Set Loss:", global_train_measures['loss'])
        print("Global Train Set Accuracy:", global_train_measures['accuracy'])
        print("Global Train Set FPR:", global_train_measures['fpr'])
        print("Global Train Set TPR:", global_train_measures['tpr'])
        print("Global Train Set BER:", global_train_measures['ber'])
        print("Global Validation Set Loss:", global_val_measures['loss'])
        print("Global Validation Set Accuracy:", global_val_measures['accuracy'])
        print("Global Validation Set FPR:", global_val_measures['fpr'])
        print("Global Validation Set TPR:", global_val_measures['tpr'])
        print("Global Validation Set BER:", global_val_measures['ber'])

    return global_train_measures, global_val_measures


def save_global_measure(measures, filename, method_name):
    os.makedirs("Experimental_results", exist_ok=True)
    os.makedirs(os.path.join("Experimental_results", method_name), exist_ok=True)
    full_path = os.path.join("Experimental_results", method_name, filename)
    file_exists = os.path.exists(full_path)

    with open(full_path, 'w', newline='') as csvfile:
        fieldnames = ['iteration',
                      f'loss',
                      f'accuracy',
                      f'fpr',
                      f'tpr',
                      f'ber']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for idx, measure in enumerate(measures):
            row = {
                'iteration': idx + 1,
                'loss': measure['loss'],
                'accuracy': measure['accuracy'],
                'fpr': measure['fpr'],
                'tpr': measure['tpr'],
                'ber': measure['ber']
            }
            writer.writerow(row)

    print(f"Saved measures to {full_path}")


def load_single_client_data(client_dir, base_path, device):
    client_path = os.path.join(base_path, client_dir)

    # Load datasets for this client
    train_df = pd.read_csv(os.path.join(client_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(client_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(client_path, "test.csv"))

    def prepare_dataset(df):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        if device is not None:
            X = torch.tensor(X, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
        return X, y

    train = prepare_dataset(train_df)
    val = prepare_dataset(val_df)
    test = prepare_dataset(test_df)

    return train, test, val


def load_client_data(base_path, device=None):
    client_data = []

    client_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and "client_" in d]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_single_client_data, client_dir, base_path, device) for client_dir in
                   client_dirs]
        for future in concurrent.futures.as_completed(futures):
            client_data.append(future.result())

    return client_data


def compute_metrics(y_true, y_pred, y_prob=None):
    accuracy = accuracy_score(y_true, y_pred)

    if y_prob is None:
        loss = hinge_loss(y_true, y_pred, labels=[-1, 1])
    else:
        loss = log_loss(y_true, y_prob, labels=[0, 1])

    confusion_vals = confusion_matrix(y_true, y_pred).ravel()
    if len(confusion_vals) == 4:
        tn_train, fp_train, fn_train, tp_train = confusion_vals
    else:
        if y_true[0] == 0:
            tn_train = confusion_vals[0]
            fp_train, fn_train, tp_train = 0, 0, 0
        else:
            tp_train = confusion_vals[0]
            tn_train, fp_train, fn_train = 0, 0, 0

    fpr = fp_train / (fp_train + tn_train) if (fp_train + tn_train) != 0 else 0
    tpr = tp_train / (tp_train + fn_train) if (tp_train + fn_train) != 0 else 0
    ber = 0.5 * ((fp_train / (fp_train + tn_train) if (fp_train + tn_train) != 0 else 0) +
                 (fn_train / (fn_train + tp_train) if (fn_train + tp_train) != 0 else 0))

    return accuracy, loss, fpr, tpr, ber

