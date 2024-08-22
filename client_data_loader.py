# To be implemented by the user
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset

fds = None
class DataClientLoader():
    def __init__(self, client_id: int,):
        self.client_id = client_id

    def load_data(self, **kwargs):
        """Load partition CIFAR10 data."""
        partition_id = kwargs['partition_id']
        num_partitions = kwargs['num_partitions']
        batch_size = kwargs['batch_size']
        # print(f"Client {self.client_id} loading partition {partition_id} of CIFAR10")
        print(f"partition_id: {partition_id}, num_partitions: {num_partitions}, batch_size: {batch_size}")
        # return
        # Only initialize `FederatedDataset` once
        global fds
        if fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
                seed=42
            )
        partition = fds.load_partition(partition_id)
        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        def apply_transforms(batch):
            """Apply transforms to the partition from FederatedDataset."""
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        return trainloader, testloader
