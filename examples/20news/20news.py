import click
import numpy as np
from torch.optim import Adam
import torch
from torch.utils.data import TensorDataset
from tensorboardX import SummaryWriter
import pickle

from ptavitm.model import train
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
    help='training batch size (default 200).',
    type=int,
    default=200
)
@click.option(
    '--epochs',
    help='number of finetune epochs (default 80).',
    type=int,
    default=80
)
@click.option(
    '--top-words',
    help='number of top words to report per topic (default 12).',
    type=int,
    default=12
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
    top_words,
    testing_mode
):
    print('Loading input data')
    # fix relative
    input_train = np.load('data/train.txt.npy', encoding='bytes')
    input_val = np.load('data/test.txt.npy', encoding='bytes')
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    reverse_vocab = {vocab[word]: word for word in vocab}
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
    # dataloader = DataLoader(
    #     ds_train,
    #     batch_size=1024,
    #     shuffle=False
    # )
    autoencoder.eval()
    # mean_batches = []
    # var_batches = []
    # for batch in dataloader:
    #     batch = batch[0]
    #     if cuda:
    #         batch = batch.cuda(non_blocking=True)
    #     _, mean, logvar = autoencoder.encode(batch)
    #     mean_batches.append(mean.detach())
    #     var_batches.append(logvar.exp().detach())
    # mean = torch.cat(mean_batches).cpu()
    # var = torch.cat(var_batches).cpu()
    decoder_weight = autoencoder.decoder.linear.weight.detach().cpu()
    topics = [
        [reverse_vocab[item.item()] for item in topic] for topic in decoder_weight.topk(top_words, dim=0)[1].t()
    ]
    for topic in topics:
        print(','.join(topic))
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
