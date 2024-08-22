# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.functional as F
# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import IidPartitioner
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose, Normalize, ToTensor

import argparse
import os
from typing import List, Tuple
from flwr.common import Metrics, ndarrays_to_parameters
from flwr_monitoring import GenericMonitoringStrategy, default_metrics, create_monitoring_tool, aggregate_fit_metrics, aggregate_evaluate_metrics
from utils import Net, get_weights
import flwr as fl
from flwr.server import ServerConfig

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--num-of-rounds", default=os.getenv('NUMBER_OF_ROUNDS', 4),
                        help="Number of rounds of federated learning")
    parser.add_argument('--server-ip', type=str,
                        default=os.getenv('SERVER_IP', '0.0.0.0'), help="Server IP address")
    parser.add_argument('--server-port', type=int,
                        default=os.getenv('SERVER_PORT', 8080), help="Server port")
    parser.add_argument('--prometheus-ip', type=str,
                        default=os.getenv('PROMETHEUS_IP', '0.0.0.0'), help="Server IP address")
    parser.add_argument('--prometheus-port', type=int,
                        default=os.getenv('PROMETHEUS_PORT', 8000), help="Server port")

    args = parser.parse_args()

    # To be provided by team
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    server_ip = args.server_ip
    server_port = args.server_port
    prometheus_ip = args.prometheus_ip
    prometheus_port = args.prometheus_port
    num_rounds = args.num_of_rounds
    
    print(f"Server IP: {server_ip}")
    print(f"Server Port: {server_port}")

    # Define strategy
    base_strategy = fl.server.strategy.FedAvg(
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        # Most important parameter: fraction of clients used for montioring
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )
    config = ServerConfig(num_rounds=num_rounds)

    # Start Flower server for four rounds of federated learning
    monitoring_tool_instance = create_monitoring_tool(
        tool_name="prometheus",
        metrics=default_metrics,
        config={"port": prometheus_port, "url": prometheus_ip}
    )

    # Wrap the base strategy with the monitoring strategy
    monitoring_strategy = GenericMonitoringStrategy(
        base_strategy, monitoring_tool_instance
    )

    server_address = f"{server_ip}:{server_port}"

    print(f"Starting Flower server at {server_ip}:{server_port}")
    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=monitoring_strategy,
    )

if __name__ == "__main__":
    main()