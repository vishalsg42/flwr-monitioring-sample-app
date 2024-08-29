import argparse
import time
import warnings
import os

import flwr as fl
from flwr.client import ClientApp, NumPyClient
import torch
from client_data_loader import DataClientLoader

from utils import Net, get_weights, set_weights, test, train

warnings.filterwarnings("ignore")

# Define Flower Client


class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(**kwargs) -> NumPyClient:
    # Create data loaders
    client_id = kwargs["client_id"]
    local_epochs = kwargs["local_epochs"]
    learning_rate = kwargs["learning_rate"]

    trainloader, valloader = DataClientLoader(client_id=client_id).load_data(
        partition_id=0, num_partitions=1, batch_size=1)

    # Create the Flower client
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate)


def main() -> None:
    # Start time of the entire script
    start_time = time.time()

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--client-id",
        type=int,
        default=os.getenv('CLIENT_ID', '0'),
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=os.getenv('LOCAL_EPOCHS', '0'),
        choices=range(0, 10),
        required=False,
        help="Specifies the number of local epochs to be used.",
    )
    parser.add_argument(
        "--learning-rate",
        type=int,
        default=os.getenv('LEARNING_RATE', '0'),
        choices=range(0, 10),
        required=False,
        help="Specifies the learning rate to be used.",
    )
    parser.add_argument(
        "--use-cuda",
        type=bool,
        default=os.getenv('USE_CUDA', 'False'),
        required=False,
        help="Set to true to use GPU. Default: False",
    )
    parser.add_argument(
        '--server-ip', type=str,
        default=os.getenv('SERVER_IP', '0.0.0.0'),
        help="Server IP address"
    )
    parser.add_argument(
        '--server-port',
        type=str,
        default=os.getenv('SERVER_PORT', '8080'),
        help="Server port"
    )

    args = parser.parse_args()

    server_ip = args.server_ip
    server_port = args.server_port
    client_id = args.client_id
    local_epochs = args.local_epochs
    learning_rate = args.learning_rate

    # trainloader, valloader = DataClientLoader(client_id=client_id).load_data(
    # partition_id=0, num_partitions=1, batch_size=1)
    # client_fn = FlowerClient(trainloader, valloader, local_epochs, learning_rate).to_client()

    # app = ClientApp(client_fn)

    # fl.client.start_client(
    #     server_address=f"{server_ip}:{server_port}",
    #     client=client_fn
    # )

    print(f"Starting Flower client at {server_ip}:{server_port}")
    fl.client.start_client(
        server_address=f"{server_ip}:{server_port}",
        client=client_fn(client_id=client_id, local_epochs=local_epochs, learning_rate=learning_rate).to_client()
    )

    # End time of the entire script
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
