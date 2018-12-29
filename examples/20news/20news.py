import click
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter
import uuid
import pickle

from ptavitm.model import train, predict
from ptavitm.vae import ProdLDA

@click.command()
@click.option(
    '--cuda',
    help='whether to use CUDA (default False).',
    type=bool,
    default=False
)
@click.option(
    '--batch-size',
    help='training batch size (default 256).',
    type=int,
    default=256
)
@click.option(
    '--epochs',
    help='number of finetune epochs (default 500).',
    type=int,
    default=20
)
@click.option(
    '--testing-mode',
    help='whether to run in testing mode (default False).',
    type=bool,
    default=False
)
def main(
    cuda,
    batch_size,
    epochs,
    testing_mode
):
    print('Loading input data')
    # fix relative
    input_train = np.load('data/train.txt.npy', encoding='bytes')
    input_val = np.load('data/test.txt.npy', encoding='bytes')
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    data_train = np.array([np.bincount(doc.astype('int'), minlength=len(vocab)) for doc in input_train if doc.sum() > 0])
    data_val = np.array([np.bincount(doc.astype('int'), minlength=len(vocab)) for doc in input_val if doc.sum() > 0])

    writer = SummaryWriter()  # create the TensorBoard object
    # callback function to call during training, uses writer from the scope

    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars('data/autoencoder', {
            'lr': lr,
            'loss': loss,
            'validation_loss': validation_loss,
        }, epoch)
    ds_train = TensorDataset(torch.from_numpy(data_train).float())
    ds_val = TensorDataset(torch.from_numpy(data_val).float())
    autoencoder = ProdLDA(
        in_dimension=len(vocab),
        hidden1_dimension=100,
        hidden2_dimension=100,
        topics=50
    )
    if cuda:
        autoencoder.cuda()
    print('Training stage.')
    ae_optimizer = Adam(autoencoder.parameters(), 0.002, betas=(0.99, 0.999))
    train(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        update_callback=None  # TODO
    )
    dataloader = DataLoader(
        ds_train,
        batch_size=1024,
        shuffle=False
    )
    autoencoder.eval()
    features = []
    for index, batch in enumerate(dataloader):
        batch[0] = batch
        if cuda:
            batch = batch.cuda(non_blocking=True)
        features.append(autoencoder.encoder(batch).detach().cpu())
    print(1)
    # if not testing_mode:
    #     writer.add_embedding(
    #         torch.cat(features),
    #         metadata=predicted,
    #         label_img=ds_train.ds.train_data.float().unsqueeze(1),  # TODO bit ugly
    #         tag='predicted'
    #     )
    #     writer.close()


if __name__ == '__main__':
    main()
