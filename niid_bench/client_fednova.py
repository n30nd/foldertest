"""Defines the client class and support functions for FedNova."""

from typing import Callable, Dict, List, OrderedDict

import flwr as fl
import torch
from flwr.common import Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from niid_bench.models import test, train_fednova


# pylint: disable=too-many-instance-attributes
class FlowerClientFedNova(fl.client.NumPyClient):
    """Flower client implementing FedNova."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
    # Trong client_fednova.py

    # Trong client_fednova.py

    # def get_parameters(self, config: Dict[str, Scalar]):
    #     """Trả về các tham số mô hình hiện tại của local."""
    #     state_dict = self.net.state_dict()
    #     # Loại bỏ các buffer không cần thiết
    #     filtered_state_dict = {k: v for k, v in state_dict.items() if 'num_batches_tracked' not in k}
    #     return [val.cpu().numpy() for val in filtered_state_dict.values()]

    # def set_parameters(self, parameters):
    #     """Thiết lập các tham số mô hình local bằng các tham số cho trước."""
    #     state_dict = self.net.state_dict()
    #     keys = [k for k in state_dict.keys() if 'num_batches_tracked' not in k]
    #     new_state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in zip(keys, parameters)})
    #     self.net.load_state_dict(new_state_dict, strict=False)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for FedNova."""
        # print("parameters", type(parameters))
        # for idx, param in enumerate(parameters):
        #     print(f"param {idx} shape: {param.shape}")
        self.set_parameters(parameters)

        a_i, g_i = train_fednova(
            self.net,
            self.trainloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
        )
        # for idx, param in enumerate(parameters):
        #     print(f"param {idx} shape: {param.shape}")
        # final_p_np = self.get_parameters({})
        g_i_np = [param.cpu().numpy() for param in g_i]
        # print("g_i_np", type(g_i_np))
        # for idx, gi in enumerate(g_i_np):
        #     print(f"g_i_np {idx} shape: {gi.shape}")
        # g_i_np = [param.cpu().detach().numpy() for param in g_i]
        return g_i_np, len(self.trainloader.dataset), {"a_i": a_i}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        loss, acc = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}


# pylint: disable=too-many-arguments
def gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    num_epochs: int,
    learning_rate: float,
    model: DictConfig,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
) -> Callable[[str], FlowerClientFedNova]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the FedNova flower clients.

    Parameters
    ----------
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    model : DictConfig
        The model configuration.
    momentum : float
        The momentum for SGD optimizer of clients
    weight_decay : float
        The weight decay for SGD optimizer of clients

    Returns
    -------
    Callable[[str], FlowerClientFedNova]
        The client function that creates the FedNova flower clients
    """

    def client_fn(cid: str) -> FlowerClientFedNova:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClientFedNova(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            momentum,
            weight_decay,
        )

    return client_fn
