import tensorflow_federated as tff

def download_and_save_federated_cifar100():
    cifar100_train, cifar100_test = tff.simulation.datasets.cifar100.load_data(cache_dir='./')
    # train_client_id = cifar100_train.client_ids
    # for client_id in train_client_id:
    #     train_data = cifar100_train.create_tf_dataset_for_client(client_id)
    #     batch_data = train_data.batch(100)
    #     batch_data = list(batch_data.as_numpy_iterator())
    #     batch_data = batch_data[0]
    #     image = batch_data['image']
    #     label = batch_data['label']
    #     print()
    # print()
    
    
"""
#with Tensorflow dependencies, you can run this python script to process the data from Tensorflow Federated locally:
python dataset.py

Before downloading, please install TFF as its official instruction:
pip install --upgrade tensorflow_federated
"""
if __name__ == "__main__":
    download_and_save_federated_cifar100()