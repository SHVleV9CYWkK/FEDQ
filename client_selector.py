import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # Store more information in the buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def flatten_parameters(parameters):
    return np.concatenate([param.detach().cpu().numpy().flatten() for param in parameters]).reshape(1, -1)


class ClientSelector:
    def __init__(self, clients, select_num, hyperparameters=None):
        random.seed(hyperparameters['seed'])
        np.random.seed(hyperparameters['seed'])
        self.enable = hyperparameters['enable']
        self.is_train = hyperparameters['is_train']
        self.type = hyperparameters['type_name']
        self.clients = clients

        if not self.enable:
            return
        self.select_num = select_num
        self.clients = clients

        if self.select_num > len(self.clients):
            raise ValueError("select_num should not be larger than the length of clients")

        if self.type == "random" or hyperparameters is None:
            print("Random selection of clients")
            self.select_func = self.__random_select
        elif self.type == "dqn" or self.type == "DQN":
            print("DQN selection of clients")
            self.select_func = self.__dqn_select

            self.epsilon_start = hyperparameters['epsilon_start']
            self.epsilon_end = hyperparameters['epsilon_end']
            self.epsilon_decay = hyperparameters['epsilon_decay']
            self.lambda_value = hyperparameters['reward_lambda_value']
            self.gamma = hyperparameters['gamma']
            self.target_accuracy = hyperparameters['target_accuracy']
            self.path = hyperparameters['model_path']
            self.device = hyperparameters['device']

            # Initialize DQN
            self.model = DQNNetwork(hyperparameters['pca_n_components'] * hyperparameters['global_models_num']
                                    , len(self.clients)).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), hyperparameters['lr'])
            self.criterion = nn.MSELoss()

            # ε-greedy parameters
            self.steps_done = 0

            # Initialize PCA
            self.pcas = [PCA(n_components=hyperparameters['pca_n_components'])] * hyperparameters['global_models_num']

            # Load model
            if not self.is_train:
                self.__load_model()

            self.buffer = ReplayBuffer(hyperparameters['buffer_size'])
            self.batch_size = hyperparameters['buffer_batch_size']

    def sample_clients(self, global_model_params_list=None, new_client_module_lists=None):
        if self.enable:
            return self.select_func(global_model_params_list, new_client_module_lists)[0]
        return self.clients

    def __random_select(self, global_model_params_list=None, new_client_module_lists=None):
        shuffled_clients = self.clients.copy()
        random.shuffle(shuffled_clients)
        selected_clients = shuffled_clients[:self.select_num]
        indices = [client.id for client in shuffled_clients]
        return selected_clients, indices

    def __dqn_select(self, global_module_list, client_module_lists):
        if global_module_list is None:
            raise ValueError("DQN selection must be passed as model parameters")

        epsilon = 0.0
        if self.is_train:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      torch.exp(torch.tensor(-1.) * self.steps_done / self.epsilon_decay)
            print("epsilon: ", epsilon)

        pca_result = np.array(self.__get_pca_result(global_module_list, client_module_lists))
        self.prepare_store_state = torch.tensor(pca_result, dtype=torch.float32).flatten().unsqueeze(0).to(
            self.device)
        self.steps_done += 1

        if self.is_train and random.random() < epsilon:
            print("Random select clients due to epsilon-greedy policy")
            selected_clients, sorted_indices = self.__random_select()
        else:
            # Get Q-values from the model
            q_values = self.model(self.prepare_store_state)
            # Select top clients based on Q-values
            _, sorted_indices = torch.sort(q_values, descending=True)
            sorted_indices = sorted_indices.squeeze().tolist()
            selected_indices = sorted_indices[:self.select_num]
            selected_clients = [self.clients[i] for i in selected_indices]

        self.last_action = sorted_indices
        return selected_clients, sorted_indices

    def update_dqn(self, new_global_module_list, client_module_lists, current_accuracy):
        if not self.enable or not self.is_train or self.type == 'random':
            return

        pca_result = np.array(self.__get_pca_result(new_global_module_list, client_module_lists))
        new_state = torch.tensor(pca_result, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)

        # Store the experience in the replay buffer
        rewards = torch.tensor(self.lambda_value * (current_accuracy - self.target_accuracy) - 1,
                               dtype=torch.float32).to(self.device)
        print("rewards:", rewards, " accuracy:", current_accuracy)
        self.buffer.push(self.prepare_store_state, self.last_action, rewards, new_state,
                         current_accuracy >= self.target_accuracy)
        print("len buffer: ", len(self.buffer))

        if len(self.buffer) < self.batch_size:
            return

        print("Training DQN")

        # Sample a batch from the replay buffer
        transitions = self.buffer.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.cat(batch_state).to(self.device)
        batch_reward = torch.stack(batch_reward).to(self.device)
        batch_next_state = torch.cat(batch_next_state).to(self.device)  # Ensure next state is on the correct device
        batch_done = torch.tensor(batch_done, dtype=torch.bool).to(self.device)  # Convert to boolean tensor

        # Compute the target Q-values
        with torch.no_grad():
            max_next_q_values = self.model(batch_next_state).max(1)[0]
            target_q_values = torch.where(batch_done, batch_reward, batch_reward + self.gamma * max_next_q_values)

        batch_action_tensor = torch.tensor(batch_action, dtype=torch.int64).to(self.device)
        batch_action_indices = batch_action_tensor.argmax(dim=1, keepdim=True)
        predicted_q_values = self.model(batch_state).gather(1, batch_action_indices).squeeze(1)

        # Compute loss
        loss = self.criterion(predicted_q_values, target_q_values)
        print("loss:", loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, datasets=""):
        if self.enable and self.is_train:
            current_time = datetime.now()
            formatted_time = current_time.strftime('%d-%m-%Y_%H-%M-%S')
            save_model_path = os.path.join("dqn_models", formatted_time+"_"+datasets)

            # Ensure the directory exists
            if not os.path.exists("dqn_models"):
                os.makedirs("dqn_models")

            torch.save(self.model.state_dict(), save_model_path)

    def __load_model(self):
        if os.path.exists(self.path):
            self.model.load_state_dict(torch.load(self.path))
            self.model.eval()
            print("Successfully loaded model")
        else:
            print(f"Model at {self.path} not found!")

    def __get_pca_result(self, global_module_params_list, client_module_params_list):
        result = [None] * len(global_module_params_list)

        for idx, module in enumerate(global_module_params_list):
            global_model_data = flatten_parameters(module.model.parameters())
            all_client_data = []
            for client_params_dict in client_module_params_list[idx]:
                client_data = flatten_parameters(client_params_dict.values())
                all_client_data.append(client_data)
            combined_data = np.vstack([global_model_data] + all_client_data)
            transformed_data = self.pcas[idx].transform(combined_data)
            global_transformed_data = transformed_data[0]
            result[idx] = global_transformed_data
        return result

    def fit_pca(self, global_module_params_list, client_module_params_list):
        if self.enable and (self.type == "dqn" or self.type == "DQN"):
            for idx, module in enumerate(global_module_params_list):
                global_model_data = flatten_parameters(module.model.parameters())
                all_client_data = []
                for client_params_dict in client_module_params_list[idx]:
                    client_data = flatten_parameters(client_params_dict.values())
                    all_client_data.append(client_data)
                combined_data = np.vstack([global_model_data] + all_client_data)
                self.pcas[idx].fit(combined_data)