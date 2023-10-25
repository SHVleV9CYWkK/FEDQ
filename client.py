import torch
import torch.nn.functional as F
from torch import nn
from module import Module, MLPModule
from utils import compute_metrics


class Client:
    def __init__(self, id, hyperparameters, train_set, test_set, val_set=None):
        self.id = id
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set
        torch.manual_seed(hyperparameters['seed'])

        module_class = MLPModule
        self.local_module: Module = module_class(hyperparameters)

    def receive_global_model(self, global_module: Module):
        self.local_module.set_parameters(global_module.get_parameters())

    def local_fit_and_upload_parameters(self):
        self.local_module.fit(self.train_set[0], self.train_set[1])
        measures = self.evaluate()
        parameters = self.local_module.get_parameters()
        train_set_n_samples = len(self.train_set[0])
        val_set_n_samples = len(self.val_set[0])
        return parameters, measures, (train_set_n_samples, val_set_n_samples)

    def evaluate(self):
        y_train_true = self.train_set[1]
        y_val_true = self.val_set[1]

        # Predict probabilities on training and validation sets
        y_train_prob = self.local_module.predict_proba(self.train_set[0], is_tensor=True, is_eval=True)
        y_val_prob = self.local_module.predict_proba(self.val_set[0], is_tensor=True, is_eval=True)

        # Convert probabilities to class labels
        y_train_pred = (y_train_prob > 0.5).cpu().numpy().astype(int)
        y_val_pred = (y_val_prob > 0.5).cpu().numpy().astype(int)

        # Compute metrics
        train_metrics = compute_metrics(y_train_true.cpu().numpy(), y_train_pred, y_train_prob.cpu().numpy())
        val_metrics = compute_metrics(y_val_true.cpu().numpy(), y_val_pred, y_val_prob.cpu().numpy())

        measures = {
            "train": {
                "accuracy": train_metrics[0],
                "loss": train_metrics[1],
                "fpr": train_metrics[2],
                "tpr": train_metrics[3],
                "ber": train_metrics[4]
            },
            "val": {
                "accuracy": val_metrics[0],
                "loss": val_metrics[1],
                "fpr": val_metrics[2],
                "tpr": val_metrics[3],
                "ber": val_metrics[4]
            }
        }

        return measures

    def evaluate_test_set(self):
        y_true = self.test_set[1]
        y_prob = self.local_module.predict_proba(self.test_set[0])

        y_pred = (y_prob > 0.5).astype(int)

        accuracy, loss, fpr, tpr, ber = compute_metrics(y_true, y_pred, y_prob)

        measures = {
            "accuracy": accuracy,
            "loss": loss,
            "fpr": fpr,
            "tpr": tpr,
            "ber": ber
        }

        test_set_n_samples = len(self.test_set[0])

        return measures, test_set_n_samples


class EMClient(Client):
    def __init__(self, id, hyperparameters, train_set, test_set, val_set=None):
        super().__init__(id, hyperparameters, train_set, test_set, val_set)
        self.n_mix_distribute = hyperparameters['models_num']
        self.device = hyperparameters['device']
        self.local_module = None

        # Initialize multiple models for each mixture distribution
        self.local_modules = [MLPModule(hyperparameters) for _ in range(self.n_mix_distribute)]
        self.posterior_probs = None
        self.learners_weights = torch.ones(self.n_mix_distribute).to(self.device) / self.n_mix_distribute

    def e_step(self):
        criterion = nn.BCELoss(reduction="none").to(self.device)
        all_losses = []
        for module in self.local_modules:
            prob = module.predict_proba(self.train_set[0], is_tensor=True, is_eval=True)
            loss = criterion(prob, self.train_set[1])
            all_losses.append(loss)
        weighted_losses = torch.stack(all_losses).T - torch.log(self.learners_weights + 1e-10)
        self.posterior_probs = F.softmax(-weighted_losses, dim=1)

    def m_step(self):
        for idx, module in enumerate(self.local_modules):
            weights = self.posterior_probs[:, idx]
            module.fit(self.train_set[0], self.train_set[1], weights)
        self.learners_weights = self.posterior_probs.mean(dim=0)

    def train_module(args):
        module, X, y, weights = args
        module.fit(X, y, weights)
        return module.get_parameters()

    def receive_global_model(self, global_module_list):
        for local_module, global_module in zip(self.local_modules, global_module_list):
            local_module.set_parameters(global_module.get_parameters())

    def local_fit_and_upload_parameters(self):
        self.e_step()
        self.m_step()

        all_parameters = [module.get_parameters() for module in self.local_modules]

        measures = self.evaluate()
        train_set_n_samples = len(self.train_set[0])
        val_set_n_samples = len(self.val_set[0])
        return all_parameters, measures, (train_set_n_samples, val_set_n_samples)

    def evaluate(self):
        y_train_true = self.train_set[1]
        y_val_true = self.val_set[1]

        # Initialize aggregated predictions
        y_train_pred_aggregated = torch.zeros_like(y_train_true, dtype=torch.float32)
        y_val_pred_aggregated = torch.zeros_like(y_val_true, dtype=torch.float32)

        # Compute weighted average predictions from all modules (models)
        for idx, module in enumerate(self.local_modules):
            weight = self.learners_weights[idx]
            y_train_pred_aggregated += weight * module.predict_proba(self.train_set[0], is_tensor=True, is_eval=True)
            y_val_pred_aggregated += weight * module.predict_proba(self.val_set[0], is_tensor=True, is_eval=True)

        # Convert aggregated probabilities to class labels
        y_train_pred = (y_train_pred_aggregated > 0.5).cpu().numpy().astype(int)
        y_val_pred = (y_val_pred_aggregated > 0.5).cpu().numpy().astype(int)

        # Compute metrics
        train_metrics = compute_metrics(y_train_true.cpu().numpy(), y_train_pred, y_train_pred_aggregated.cpu().numpy())
        val_metrics = compute_metrics(y_val_true.cpu().numpy(), y_val_pred, y_val_pred_aggregated.cpu().numpy())

        measures = {
            "train": {
                "accuracy": train_metrics[0],
                "loss": train_metrics[1],
                "fpr": train_metrics[2],
                "tpr": train_metrics[3],
                "ber": train_metrics[4]
            },
            "val": {
                "accuracy": val_metrics[0],
                "loss": val_metrics[1],
                "fpr": val_metrics[2],
                "tpr": val_metrics[3],
                "ber": val_metrics[4]
            }
        }

        return measures

    def evaluate_test_set(self):
        y_true = self.test_set[1]

        y_pred_aggregated = torch.zeros_like(y_true, dtype=torch.float32)

        for idx, module in enumerate(self.local_modules):
            weight = self.learners_weights[idx]
            y_pred_aggregated += weight * torch.tensor(module.predict_proba(self.test_set[0]))

        y_pred = (y_pred_aggregated > 0.5).numpy().astype(int)

        accuracy, loss, fpr, tpr, ber = compute_metrics(y_true, y_pred, y_pred_aggregated.numpy())

        measures = {
            "accuracy": accuracy,
            "loss": loss,
            "fpr": fpr,
            "tpr": tpr,
            "ber": ber
        }

        test_set_n_samples = len(self.test_set[0])

        return measures, test_set_n_samples
