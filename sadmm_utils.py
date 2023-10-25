from utils import compute_metrics
import numpy as np

def get_value(data, node_id, neighbour_id):
    try:
        neighbour_data = data[(node_id, neighbour_id)]
        val = neighbour_data[0]
    except KeyError:
        neighbour_data = data[(neighbour_id, node_id)]
        val = neighbour_data[1]

    return val


def calculate_consensus(z, edges):
    num_edges = len(z)
    cons = 0
    for edge in edges:
        z_edge = z[edge]
        if np.all(z_edge[0] == z_edge[1]):
            cons += 1
    return cons / float(num_edges)


def calculate_accuracy(x, num_nodes, datasets_test, iterate_num=None):
    total_correct_preds = 0
    num_total_test_samples = 0

    for node_id in range(num_nodes):
        X_test, y_test = datasets_test[node_id]

        num_test_samples = y_test.shape[0]
        num_total_test_samples += num_test_samples
        a = np.array(x[:, node_id])
        a = a.reshape(1, a.shape[0])
        y_pred = np.sign(np.dot(X_test, a.T)).flatten()
        correct_preds = int(np.sum(np.abs(y_pred + y_test)) / 2)
        total_correct_preds += correct_preds

    if iterate_num is None:
        global_correct_preds = total_correct_preds / float(num_total_test_samples)
        print("Global Accuracy: " + str(global_correct_preds))
        return global_correct_preds
    else:
        global_acc = total_correct_preds / float(num_total_test_samples)
        print("Iteration: " + str(iterate_num) + ", Accuracy: " + str(global_acc))


def stable_sigmoid(x):
    # Clip values for stability
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def calculate_metrics(x, num_nodes, datasets_test):
    total_samples = 0
    total_accuracy = 0
    total_loss = 0
    total_fpr = 0
    total_tpr = 0
    total_ber = 0

    for node_id in range(num_nodes):
        X_test, y_test = datasets_test[node_id]

        a = np.array(x[:, node_id])
        a = a.reshape(1, a.shape[0])
        y_pred_dec_func = np.dot(X_test, a.T).flatten()
        y_pred = np.sign(y_pred_dec_func)
        y_pred_prob = stable_sigmoid(y_pred_dec_func)

        # Convert -1 labels to 0 for computing metrics
        y_test = [(label + 1) // 2 for label in y_test]
        y_pred = [(label + 1) // 2 for label in y_pred]

        accuracy, loss, fpr, tpr, ber = compute_metrics(y_test, y_pred, y_pred_prob)

        num_samples = len(y_test)
        total_samples += num_samples

        total_accuracy += accuracy
        total_loss += loss
        total_fpr += fpr
        total_tpr += tpr
        total_ber += ber

    weighted_accuracy = total_accuracy / num_nodes
    weighted_loss = total_loss / num_nodes
    weighted_fpr = total_fpr / num_nodes
    weighted_tpr = total_tpr / num_nodes
    weighted_ber = total_ber / num_nodes

    return {"accuracy": weighted_accuracy, "loss": weighted_loss, "fpr": weighted_fpr, "tpr": weighted_tpr, "ber": weighted_ber}

