import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# def get_mnist(data_path: str = "./data"):
#     """Download MNIST and apply minimal transformation."""

#     tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

#     trainset = MNIST(data_path, train=True, download=True, transform=tr)
#     testset = MNIST(data_path, train=False, download=True, transform=tr)

#     return trainset, testset


# def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
#     """Download MNIST and generate IID partitions."""

#     # download MNIST in case it's not already in the system
#     trainset, testset = get_mnist()

#     # split trainset into `num_partitions` trainsets (one per client)
#     # figure out number of training examples per partition
#     num_images = len(trainset) // num_partitions

#     # a list of partition lenghts (all partitions are of equal size)
#     partition_len = [num_images] * num_partitions

#     # split randomly. This returns a list of trainsets, each with `num_images` training examples
#     # Note this is the simplest way of splitting this dataset. A more realistic (but more challenging) partitioning
#     # would induce heterogeneity in the partitions in the form of for example: each client getting a different
#     # amount of training examples, each client having a different distribution over the labels (maybe even some
#     # clients not having a single training example for certain classes). If you are curious, you can check online
#     # for Dirichlet (LDA) or pathological dataset partitioning in FL. A place to start is: https://arxiv.org/abs/1909.06335
#     trainsets = random_split(
#         trainset, partition_len, torch.Generator().manual_seed(2023)
#     )

#     # create dataloaders with train+val support
#     trainloaders = []
#     valloaders = []
#     # for each train set, let's put aside some training examples for validation
#     for trainset_ in trainsets:
#         num_total = len(trainset_)
#         num_val = int(val_ratio * num_total)
#         num_train = num_total - num_val

#         for_train, for_val = random_split(
#             trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
#         )

#         # construct data loaders and append to their respective list.
#         # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
#         trainloaders.append(
#             DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
#         )
#         valloaders.append(
#             DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
#         )

#     # We leave the test set intact (i.e. we don't partition it)
#     # This test set will be left on the server side and we'll be used to evaluate the
#     # performance of the global model after each round.
#     # Please note that a more realistic setting would instead use a validation set on the server for
#     # this purpose and only use the testset after the final round.
#     # Also, in some settings (specially outside simulation) it might not be feasible to construct a validation
#     # set on the server side, therefore evaluating the global model can only be done by the clients. (see the comment
#     # in main.py above the strategy definition for more details on this)
#     testloader = DataLoader(testset, batch_size=128)

#     return trainloaders, valloaders, testloader


def get_custom_dataset(data_path: str = "./data"):
    """Load custom dataset and apply transformations."""
    transform = Compose([
        Resize((100, 100)),
        Grayscale(num_output_channels=1),
        ToTensor()
    ])
    trainset = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
    testset = ImageFolder(os.path.join(data_path, 'test'), transform=transform)
    return trainset, testset

def prepare_dataset_for_centralized_train(batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_custom_dataset()
    # Split trainset into trainset and valset
    num_train = int((1 - val_ratio) * len(trainset))
    num_val = len(trainset) - num_train
    trainset, valset = random_split(trainset, [num_train, num_val], torch.Generator().manual_seed(2024))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=6)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6)

    print(f'Number of train samples: {len(trainset)}, val samples: {len(valset)}, test samples: {len(testloader.dataset)}')
    return trainloader, valloader, testloader


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1, alpha: float = 100):
    """Load custom dataset and generate non-IID partitions using Dirichlet distribution."""
    trainset, testset = get_custom_dataset()
    
    # Split trainset into trainset and valset
    num_train = int((1 - val_ratio) * len(trainset))
    num_val = len(trainset) - num_train
    trainset, valset = random_split(trainset, [num_train, num_val], torch.Generator().manual_seed(2024))
    
    # Get labels for the entire trainset
    train_labels = np.array([trainset.dataset.targets[i] for i in trainset.indices])
    
    # Generate Dirichlet distribution for each class
    class_indices = [np.where(train_labels == i)[0] for i in range(len(np.unique(train_labels)))]
    partition_indices = [[] for _ in range(num_partitions)]
    
    for class_idx in class_indices:
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_partitions))
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        class_partitions = np.split(class_idx, proportions)
        for i in range(num_partitions):
            partition_indices[i].extend(class_partitions[i])
    
    # Create Subsets for each partition
    trainsets = [Subset(trainset.dataset, indices) for indices in partition_indices]
    
    # Split valset into partitions
    partition_len_val = [len(valset) // num_partitions] * num_partitions
    for i in range(len(valset) % num_partitions):
        partition_len_val[i] += 1

    valsets = random_split(valset, partition_len_val, torch.Generator().manual_seed(2023))
    
    # Create DataLoaders for each partition
    trainloaders = [DataLoader(ts, batch_size=batch_size, shuffle=True, num_workers=2) for ts in trainsets]
    valloaders = [DataLoader(vs, batch_size=batch_size, shuffle=False, num_workers=2) for vs in valsets]
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Calculate class distribution for each partition in trainloaders
    class_distributions = []
    for i, trainloader in enumerate(trainloaders):
        class_counts = Counter()
        for _, labels in trainloader:
            class_counts.update(labels.numpy())
        class_distributions.append(class_counts)
        print(f'Partition {i} class distribution: {dict(class_counts)}')

    # Plot class distribution
    partitions = range(num_partitions)
    class_0_counts = [class_distributions[i][0] for i in partitions]
    class_1_counts = [class_distributions[i][1] for i in partitions]

    bar_width = 0.5
    plt.figure(figsize=(12, 8))
    plt.bar(partitions, class_0_counts, bar_width, label='Class 0', color='blue')
    plt.bar(partitions, class_1_counts, bar_width, bottom=class_0_counts, label='Class 1', color='red')
    plt.xlabel('Partition')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Each Partition')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f'Number of train samples: {len(trainset)}, val samples: {len(valset)}, test samples: {len(testloader.dataset)}')
    return trainloaders, valloaders, testloader

def prepare_partitioned_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1, num_labels_each_party: int = 1):
    """Load custom dataset and generate partitions where each party has a fixed number of labels."""
    trainset, testset = get_custom_dataset()  # Load datasets

    # Split the trainset into trainset and valset based on the validation ratio
    num_train = int((1 - val_ratio) * len(trainset))
    num_val = len(trainset) - num_train
    trainset, valset = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(2023))

    # Get labels for the entire trainset
    train_labels = np.array([trainset.dataset.targets[i] for i in trainset.indices])

    # Define partitions: each party has k labels
    num_labels = len(np.unique(train_labels))  # Assuming labels are 0 and 1 for binary classification
    times = [0 for i in range(num_labels)]
    contain = []
    #Phan label cho cac client
    for i in range(num_partitions):
        current = [i%num_labels]
        times[i%num_labels] += 1
        if num_labels_each_party > 1:
            current.append(1-i%num_labels)
            times[1-i%num_labels] += 1
        contain.append(current)
    print(times)
    print(contain)
    # Create Subsets for each partition

    partition_indices = [[] for _ in range(num_partitions)]
    for i in range(num_labels):
        idx_i = np.where(train_labels == i)[0]  # Get indices of label i in train_labels
        idx_i = [trainset.indices[j] for j in idx_i]  # Convert indices to indices in trainset
        # #print label of idx_i
        # print("Label of idx: ", i)
        # for j in range(len(idx_i)):
        #     idx_in_dataset = trainset.indices[idx_i[j]]
        #     print(trainset.dataset.targets[idx_in_dataset])
        np.random.shuffle(idx_i)
        split = np.array_split(idx_i, times[i])
        ids = 0
        for j in range(num_partitions):
            if i in contain[j]:
                partition_indices[j].extend(split[ids])
                ids += 1
    
    trainsets = [Subset(trainset.dataset, indices) for indices in partition_indices]

    # #print label of client 0
    # print("Client 0")
    # for i in range(len(trainsets[0])):
    #     print(trainsets[0][i][1])

    # Split valset into partitions
    partition_len_val = [len(valset) // num_partitions] * num_partitions
    for i in range(len(valset) % num_partitions):
        partition_len_val[i] += 1
    
    valsets = random_split(valset, partition_len_val, generator=torch.Generator().manual_seed(2023))

    # Create DataLoaders for each partition
    trainloaders = [DataLoader(ts, batch_size=batch_size, shuffle=True, num_workers=2) for ts in trainsets]
    valloaders = [DataLoader(vs, batch_size=batch_size, shuffle=False, num_workers=2) for vs in valsets]
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Calculate class distribution for each partition in trainloaders
    class_distributions = []
    for i, trainloader in enumerate(trainloaders):
        class_counts = Counter()
        for _, labels in trainloader:
            class_counts.update(labels.numpy())
        class_distributions.append(class_counts)
        print(f'Partition {i} class distribution: {dict(class_counts)}')
    # Plot class distribution
    partitions = range(num_partitions)
    class_0_counts = [class_distributions[i][0] for i in partitions]
    class_1_counts = [class_distributions[i][1] for i in partitions]

    bar_width = 0.5
    plt.figure(figsize=(12, 8))
    plt.bar(partitions, class_0_counts, bar_width, label='Class 0', color='blue')
    plt.bar(partitions, class_1_counts, bar_width, bottom=class_0_counts, label='Class 1', color='red')
    plt.xlabel('Partition')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Each Partition')
    plt.legend()
    plt.grid(True)
    # plt.show()

    #  Lưu đồ thị vào thư mục running_outputs với tên data_partition
    output_dir = 'running_outputs'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'data_partition.png'))
    plt.close()



    print(f'Number of train samples: {len(trainset)}, val samples: {len(valset)}, test samples: {len(testloader.dataset)}')
    return trainloaders, valloaders, testloader

def prepare_imbalance_label_dirichlet(num_partitions: int, batch_size: int, val_ratio: float = 0.1, beta: float = 0.5):
    """Load custom dataset and generate partitions where each party has a fixed number of labels."""
    trainset, testset = get_custom_dataset()  # Load datasets

    # Split the trainset into trainset and valset based on the validation ratio
    num_train = int((1 - val_ratio) * len(trainset))
    num_val = len(trainset) - num_train
    trainset, valset = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(2023))

    # Get labels for the entire trainset
    train_labels = np.array([trainset.dataset.targets[i] for i in trainset.indices])

    # Define partitions: each party has k labels
    num_labels = len(np.unique(train_labels))  # Assuming labels are 0 and 1 for binary classification
    min_size = 0
    min_require_size = 20

    N = len(trainset)


    while(min_size < min_require_size):
        partition_indices = [[] for _ in range(num_partitions)]
        for label in range(num_labels):
            idx_label = np.where(train_labels == label)[0]
            idx_label = [trainset.indices[j] for j in idx_label]
            np.random.shuffle(idx_label)

            proportions = np.random.dirichlet(np.repeat(beta, num_partitions))
            # proportions = np.array( [p * len(idx_j) < N/num_partitions] for p, idx_j in zip(proportions, partition_indices))
            proportions = np.array([p if p * len(idx_j) < N / num_partitions else 0 for p, idx_j in zip(proportions, partition_indices)])

            proportions = proportions / np.sum(proportions)
            proportions = (np.cumsum(proportions) * len(idx_label)).astype(int)[:-1]

            partition_indices = [idx_j + idx.tolist() for idx_j, idx in zip(partition_indices, np.split(idx_label, proportions))]
            min_size = min([len(idx_j) for idx_j in partition_indices])
        
    trainsets = [Subset(trainset.dataset, indices) for indices in partition_indices]

    partition_len_val = [len(valset) // num_partitions] * num_partitions
    for i in range(len(valset) % num_partitions):
        partition_len_val[i] += 1
    
    valsets = random_split(valset, partition_len_val, generator=torch.Generator().manual_seed(2023))

    trainloaders = [DataLoader(ts, batch_size=batch_size, shuffle=True, num_workers=2) for ts in trainsets]
    valloaders = [DataLoader(vs, batch_size=batch_size, shuffle=False, num_workers=2) for vs in valsets]
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_distributions = []
    for i, trainloader in enumerate(trainloaders):
        class_counts = Counter()
        for _, labels in trainloader:
            class_counts.update(labels.numpy())
        class_distributions.append(class_counts)
        print(f'Partition {i} class distribution: {dict(class_counts)}')
    # Plot class distribution
    partitions = range(num_partitions)
    class_0_counts = [class_distributions[i][0] for i in partitions]
    class_1_counts = [class_distributions[i][1] for i in partitions]

    bar_width = 0.5
    plt.figure(figsize=(12, 8))
    plt.bar(partitions, class_0_counts, bar_width, label='Class 0', color='blue')
    plt.bar(partitions, class_1_counts, bar_width, bottom=class_0_counts, label='Class 1', color='red')
    plt.xlabel('Partition')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Each Partition')
    plt.legend()
    plt.grid(True)
    #  Lưu đồ thị vào thư mục running_outputs với tên data_partition
    output_dir = 'running_outputs'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'data_partition.png'))
    plt.close()

    print(f'Number of train samples: {len(trainset)}, val samples: {len(valset)}, test samples: {len(testloader.dataset)}')

    return trainloaders, valloaders, testloader



def apply_gaussian_noise(tensor, std_dev):
    noise = torch.randn_like(tensor) * std_dev
    return tensor + noise
def prepare_noise_based_imbalance(num_partitions: int, batch_size: int, val_ratio: float = 0.1, sigma: float = 0.05):
    """
    Chia du lieu ngau nhien va deu cho cac ben, sau do them noise vao cac ben
    moi ben i co noise khac nhau Gauss(0, sigma*i/N)
    """
    trainset, testset = get_custom_dataset()
    num_train = int((1 - val_ratio) * len(trainset))
    num_val = len(trainset) - num_train
    trainset, valset = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(2023))

    indices = trainset.indices

    np.random.shuffle(indices)

    partition_indices = np.array_split(indices, num_partitions)

    train_partitions = []

    for i, part_indices in enumerate(partition_indices):
        partition_std_dev = sigma * (i + 1) / num_partitions
        partition_set = Subset(trainset.dataset, part_indices)
        
        noisy_samples = [apply_gaussian_noise(sample[0], partition_std_dev) for sample in partition_set]
        noisy_dataset = [(noisy_samples[j], trainset.dataset[part_indices[j]][1]) for j in range(len(part_indices))]
        # train_partitions.append((noisy_samples, [sample[1] for sample in partition_set]))
        train_partitions.append(noisy_dataset)
    trainloaders = [DataLoader(train_partitions[i], batch_size=batch_size, shuffle=True, num_workers=2) for i in range(num_partitions)]
    partition_len_val = [len(valset) // num_partitions] * num_partitions
    for i in range(len(valset) % num_partitions):
        partition_len_val[i] += 1
    
    valsets = random_split(valset, partition_len_val, generator=torch.Generator().manual_seed(2023))
    valloaders = [DataLoader(vs, batch_size=batch_size, shuffle=False, num_workers=2) for vs in valsets]
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloaders, valloaders, testloader


def prepare_quantity_skew_dirichlet(num_partitions: int, batch_size: int, val_ratio: float = 0.1, beta: float = 10):
    trainset, testset = get_custom_dataset()
    num_train = int((1 - val_ratio) * len(trainset))
    num_val = len(trainset) - num_train
    trainset, valset = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(2023))

    all_indices = trainset.indices

    proportions = np.random.dirichlet(np.repeat(beta, num_partitions))
    proportions = (np.cumsum(proportions) * len(all_indices)).astype(int)[:-1]

    partition_indices = np.split(all_indices, proportions)

    print('Partition sizes:', [len(partition) for partition in partition_indices])

    trainsets = [Subset(trainset.dataset, indices) for indices in partition_indices]

    partition_len_val = [len(valset) // num_partitions] * num_partitions
    for i in range(len(valset) % num_partitions):
        partition_len_val[i] += 1
    
    valsets = random_split(valset, partition_len_val, generator=torch.Generator().manual_seed(2023))

    trainloaders = [DataLoader(ts, batch_size=batch_size, shuffle=True, num_workers=2) for ts in trainsets]
    valloaders = [DataLoader(vs, batch_size=batch_size, shuffle=False, num_workers=2) for vs in valsets]
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_distributions = []
    for i, trainloader in enumerate(trainloaders):
        class_counts = Counter()
        for _, labels in trainloader:
            class_counts.update(labels.numpy())
        class_distributions.append(class_counts)
        print(f'Partition {i} class distribution: {dict(class_counts)}')
    
    partitions = range(num_partitions)
    class_0_counts = [class_distributions[i][0] for i in partitions]
    class_1_counts = [class_distributions[i][1] for i in partitions]

    bar_width = 0.5
    plt.figure(figsize=(12, 8))
    plt.bar(partitions, class_0_counts, bar_width, label='Class 0', color='blue')
    plt.bar(partitions, class_1_counts, bar_width, bottom=class_0_counts, label='Class 1', color='red')
    plt.xlabel('Partition')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Each Partition')
    plt.legend()
    plt.grid(True)
    # plt.show()
    #  Lưu đồ thị vào thư mục running_outputs với tên data_partition
    output_dir = 'running_outputs'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'data_partition.png'))
    plt.close()

    print(f'Number of train samples: {len(trainset)}, val samples: {len(valset)}, test samples: {len(testloader.dataset)}')

    return trainloaders, valloaders, testloader