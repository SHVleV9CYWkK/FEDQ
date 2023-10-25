from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from utils import *


def add_gaussian_noise(data, columns, mean=0, std_dev=1.0):
    data_noisy = data.copy()
    for col in columns:
        noise = np.random.normal(mean, std_dev, data[:, col].shape)
        data_noisy[:, col] += noise
    return data_noisy


def add_flip_noise(data, label_column, flip_prob):
    flip_indices = np.random.rand(len(data)) < flip_prob
    flipped_labels = 1 - data.loc[flip_indices, label_column]
    data.loc[flip_indices, label_column] = flipped_labels
    return data


def preprocess_data(dataset, dataset_name):
    print("Processing raw data")
    all_data = None
    if dataset_name == "unsw-nb15":
        all_data = dataset[dataset[3] != '0x20205321']

        all_data = all_data.drop(all_data.columns[47], axis=1)

        all_data[1].replace({'0x000b': 11, '0x000c': 12, '-': 0}, inplace=True)
        all_data[3].replace({'0xc0a8': 49320, '-': 0, '0xcc09': 52233}, inplace=True)
        all_data[39].replace({' ': 0}, inplace=True)
        all_data[1] = all_data[1].astype(int)
        all_data[3] = all_data[3].astype(int)
        all_data[39] = all_data[39].astype(int)

        all_data[37] = all_data[37].fillna(0)
        all_data[38] = all_data[38].fillna(0)
        all_data[37] = all_data[37].astype(int)
        all_data[38] = all_data[38].astype(int)

        all_data[0] = all_data[0].apply(ip_to_int)
        all_data[2] = all_data[2].apply(ip_to_int)

    elif dataset_name == "n-baiot":
        all_data = dataset.drop(columns=[0])

    return all_data


def encode_and_scale(dataset_name, train_data, val_data, test_data):
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    X_val = val_data.iloc[:, :-1]
    y_val = val_data.iloc[:, -1]

    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    noisy_columns = []
    train_scaled = None
    val_scaled = None
    test_scaled = None

    if dataset_name == "unsw-nb15":
        columns_to_encode = [4, 5, 13]
        encoder = OrdinalEncoder(cols=columns_to_encode)
        encoder.fit(X_train, y_train)

        train_encoded = encoder.transform(X_train)
        val_encoded = encoder.transform(X_val)
        test_encoded = encoder.transform(X_test)

        scaler = MinMaxScaler()
        scaler.fit(train_encoded)

        train_scaled = scaler.transform(train_encoded)
        val_scaled = scaler.transform(val_encoded)
        test_scaled = scaler.transform(test_encoded)

        noisy_columns = [6, 14, 15, 26, 27, 30, 31]

    elif dataset_name == "n-baiot":
        scaler = MinMaxScaler()
        scaler.fit(X_train)

        train_scaled = scaler.transform(X_train)
        val_scaled = scaler.transform(X_val)
        test_scaled = scaler.transform(X_test)

        noisy_columns = list(range(train_scaled.shape[1]))

    train_scaled = add_gaussian_noise(train_scaled, noisy_columns, mean=0, std_dev=0.1)
    val_scaled = add_gaussian_noise(val_scaled, noisy_columns, mean=0, std_dev=0.1)
    test_scaled = add_gaussian_noise(test_scaled, noisy_columns, mean=0, std_dev=0.1)

    train_final = pd.concat([pd.DataFrame(train_scaled, columns=X_train.columns), y_train.reset_index(drop=True)],
                            axis=1)
    val_final = pd.concat([pd.DataFrame(val_scaled, columns=X_val.columns), y_val.reset_index(drop=True)], axis=1)
    test_final = pd.concat([pd.DataFrame(test_scaled, columns=X_test.columns), y_test.reset_index(drop=True)],
                           axis=1)

    train_final = add_flip_noise(train_final, train_final.columns[-1], 0.05)
    # val_final = add_flip_noise(val_final, val_final.columns[-1], 0.05)
    # test_final = add_flip_noise(test_final, test_final.columns[-1], 0.05)

    return train_final, val_final, test_final


def check_clients_have_data(clients_data_indices):
    all_clients_have_data = True
    for i, client_indices in enumerate(clients_data_indices):
        if len(client_indices) < 3:
            print(f"Client {i} has no data!")
            all_clients_have_data = False

    if all_clients_have_data:
        print("All clients have been allocated data!")
        return True
    else:
        return False


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    dataset_name = args.dataset
    dataset_path = os.path.join("../", dataset_name, "raw", "data_set")
    dataset = read_dataset(dataset_path)

    dataset = preprocess_data(dataset, dataset_name)

    labels = dataset.iloc[:, -1].values
    n_classes = len(np.unique(labels))
    if args.is_split:
        clients_data_indices = split_dataset_by_labels(dataset, n_classes, args.n_clients, args.n_components,
                                                       args.alpha,
                                                       seed=args.seed)

        splited_path = os.path.join("../", dataset_name, "split")
        os.makedirs(splited_path, exist_ok=True)

        print("Processing clients")
        pbar = tqdm(total=100)
        pbar.reset()
        for client_id, indices in enumerate(clients_data_indices):
            client_data = dataset.iloc[indices]
            class_counts = client_data.iloc[:, -1].value_counts()
            train_val_data, test_data = train_test_split(client_data, test_size=0.2, shuffle=True, random_state=args.seed)
            train_data, val_data = train_test_split(train_val_data, test_size=0.2, shuffle=True, random_state=args.seed)
            train_data, val_data, test_data = encode_and_scale(dataset_name, train_data, val_data, test_data)
            client_dir = os.path.join(splited_path, f"client_{client_id}")
            os.makedirs(client_dir, exist_ok=True)

            train_data.to_csv(os.path.join(client_dir, "train.csv"), index=False)
            val_data.to_csv(os.path.join(client_dir, "val.csv"), index=False)
            test_data.to_csv(os.path.join(client_dir, "test.csv"), index=False)

            pbar.update(1)
    else:
        print("Generating training set, validation set and test set")
        train_val_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=args.seed)
        train_data, val_data = train_test_split(train_val_data, test_size=0.2, shuffle=True, random_state=args.seed)
        train_data, val_data, test_data = encode_and_scale(dataset_name, train_data, val_data, test_data)

        print("Saving")
        save_dir = os.path.join("../", dataset_name, 'unsplit')
        os.makedirs(save_dir, exist_ok=True)
        train_data.to_csv(os.path.join(save_dir, "train.csv"), index=False)
        val_data.to_csv(os.path.join(save_dir, "val.csv"), index=False)
        test_data.to_csv(os.path.join(save_dir, "test.csv"), index=False)

    print("Complete data processing")
