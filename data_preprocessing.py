import os
import numpy as np
import shutil

def partition_data(train_dir, num_clients):
    # Create directories for partitioned data
    base_dir = os.path.dirname(train_dir)
    partition_dirs = []
    
    for i in range(num_clients):
        partition_dir = os.path.join(base_dir, f'client_{i}')
        if not os.path.exists(partition_dir):
            os.makedirs(partition_dir)
        else:
            # Clear the existing directory to avoid mixing old and new data
            shutil.rmtree(partition_dir)
            os.makedirs(partition_dir)
        partition_dirs.append(partition_dir)
    
    # Partition the data for each client
    for class_dir in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_dir)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            images = [os.path.join(class_path, img) for img in images]
            client_splits = np.array_split(images, num_clients)
            
            for i, client_split in enumerate(client_splits):
                client_class_dir = os.path.join(partition_dirs[i], class_dir)
                if not os.path.exists(client_class_dir):
                    os.makedirs(client_class_dir)
                for img_path in client_split:
                    shutil.copy(img_path, client_class_dir)

if __name__ == "__main__":
    # Path to the training directory
    train_dir = 'C:\\Users\\sneha\\pfl\\chest_xray\\train'
    
    # Number of clients
    num_clients = 3  # Example: 3 clients

    # Partition the data
    partition_data(train_dir, num_clients)
