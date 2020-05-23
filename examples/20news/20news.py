import click
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import Dense2Corpus
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
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=False
)
@click.option(
    "--batch-size", help="training batch size (default 200).", type=int, default=200
)
@click.option("--epochs", help="number of epochs (default 80).", type=int, default=80)
@click.option(
    "--top-words",
    help="number of top words to report per topic (default 12).",
    type=int,
    default=12,
)
@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
)
def main(
    cuda, batch_size, epochs, top_words, testing_mode,
):
    print("Loading input data")
    # TODO fix relative paths
    input_train = np.load("data/train.txt.npy", encoding="bytes")
    input_val = np.load("data/test.txt.npy", encoding="bytes")
    with open("data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    reverse_vocab = {vocab[word]: word for word in vocab}
    indexed_vocab = [reverse_vocab[index] for index in range(len(reverse_vocab))]
    data_train = np.array(
        [
            np.bincount(doc.astype("int"), minlength=len(vocab))
            for doc in input_train
            if doc.sum() > 0
        ]
    )
    data_val = np.array(
        [
            np.bincount(doc.astype("int"), minlength=len(vocab))
            for doc in input_val
            if doc.sum() > 0
        ]
    )
    corpus = Dense2Corpus(data_train, documents_columns=False)
    writer = SummaryWriter()  # create the TensorBoard object

    # callback function to call during training, uses writer from the scope
    def training_callback(autoencoder, epoch, lr, loss, perplexity):
        decoder_weight = autoencoder.decoder.linear.weight.detach().cpu()
        topics = [
            [reverse_vocab[item.item()] for item in topic]
            for topic in decoder_weight.topk(top_words, dim=0)[1].t()
        ]
        cm = CoherenceModel(
            topics=topics,
            corpus=corpus,
            dictionary=Dictionary.from_corpus(corpus, reverse_vocab),
            coherence="u_mass",
        )
        coherence = cm.get_coherence()
        coherences = cm.get_coherence_per_topic()
        for index, topic in enumerate(topics):
            print(str(index) + ":" + str(coherences[index]) + ":" + ",".join(topic))
        print(coherence)
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, "perplexity": perplexity, "coherence": coherence,},
            global_step=epoch,
        )

    ds_train = TensorDataset(torch.from_numpy(data_train).float())
    ds_val = TensorDataset(torch.from_numpy(data_val).float())
    autoencoder = ProdLDA(
        in_dimension=len(vocab), hidden1_dimension=100, hidden2_dimension=100, topics=50
    )
    if cuda:
        autoencoder.cuda()
    print("Training stage.")
    ae_optimizer = Adam(autoencoder.parameters(), 0.001, betas=(0.99, 0.999))
    train(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        update_callback=training_callback,
    )
    autoencoder.eval()
    decoder_weight = autoencoder.decoder.linear.weight.detach().cpu()
    topics = [
        [reverse_vocab[item.item()] for item in topic]
        for topic in decoder_weight.topk(top_words, dim=0)[1].t()
    ]
    cm = CoherenceModel(
        topics=topics,
        corpus=corpus,
        dictionary=Dictionary.from_corpus(corpus, reverse_vocab),
        coherence="u_mass",
    )
    coherence = cm.get_coherence()
    coherences = cm.get_coherence_per_topic()
    for index, topic in enumerate(topics):
        print(str(index) + ":" + str(coherences[index]) + ":" + ",".join(topic))
    print(coherence)
    if not testing_mode:
        writer.add_embedding(
            autoencoder.encoder.linear1.weight.detach().cpu().t(),
            metadata=indexed_vocab,
            tag="feature_embeddings",
        )
    writer.close()


if __name__ == "__main__":
    main()
