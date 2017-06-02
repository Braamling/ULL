from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.init as nninit
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from fuel.datasets import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.transformers import AgnosticSourcewiseTransformer
from pycrayon import CrayonClient
from collections import Counter

LOG2PI = np.log(2 * np.pi)
EPSILON = 1e-8


class VAE(nn.Module):
    def __init__(self,
                 num_frames=29,
                 vector_size=600,
                 dist='cat',
                 temperature=1e-2):

        super().__init__()

        self.embedding_size = 100
        self.dist = dist
        self.temperature = temperature

        self.fc1 = nn.Linear(vector_size, 300)
        self.fc11 = nn.Linear(300, 150)
        self.fc21 = nn.Linear(150, self.embedding_size)
        self.fc22 = nn.Linear(150, self.embedding_size)

        nninit.xavier_uniform(self.fc1.weight)
        nninit.xavier_uniform(self.fc11.weight)
        nninit.xavier_uniform(self.fc21.weight)
        nninit.xavier_uniform(self.fc22.weight)

        self.fc3 = nn.Linear(self.embedding_size, 150)
        self.fc31 = nn.Linear(150, 300)
        self.fc41 = nn.Linear(300, vector_size)
        self.fc42 = nn.Linear(300, vector_size)

        nninit.xavier_uniform(self.fc3.weight)
        nninit.xavier_uniform(self.fc31.weight)
        nninit.xavier_uniform(self.fc41.weight)
        nninit.xavier_uniform(self.fc42.weight)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc11(h))

        if self.dist == 'norm':
            return self.fc21(h), self.fc22(h)
        else:
            alpha = self.fc21(h).exp()
            alpha = alpha / torch.sum(alpha, dim=1).expand_as(alpha)
            return alpha, None

    def reparametrize(self, mu, logvar):

        if self.dist == 'norm':
            std = logvar.mul(0.5).exp_()
            if args.cuda:
                eps = torch.cuda.FloatTensor(std.size()).normal_()
            else:
                eps = torch.FloatTensor(std.size()).normal_()
            eps = Variable(eps)
            return eps.mul(std).add_(mu)
        elif self.dist == 'cat':
            unif = torch.rand(mu.size())
            if args.cuda:
                unif = unif.cuda()
            unif = unif * (1 - 1e-7) + EPSILON
            u = unif.cpu().numpy()
            gumbel = Variable(-torch.log(-torch.log(unif)))
            logit = (torch.log(mu + EPSILON) + gumbel)
            logit = logit / self.temperature
            z = F.softmax(logit)
            return z

    def decode(self, z):
        h = self.relu(self.fc3(z))
        h = self.relu(self.fc31(h))
        mu, logvar = self.fc41(h), self.fc42(h)
        return mu, logvar

    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z = self.reparametrize(mu_z, logvar_z)
        mu_x, logvar_x = self.decode(z)
        return mu_x, logvar_x, mu_z, logvar_z


def loss_function(mu_x, logvar_x, x, mu_z, logvar_z, dist, K):
    LL_element = ((x - mu_x)**2 / logvar_x.exp() + logvar_x + LOG2PI)
    LL = torch.sum(LL_element).mul_(0.5)

    if dist == 'norm':

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu_z.pow(2).add_(
            logvar_z.exp()).mul_(-1).add_(1).add_(logvar_z)
        KLD = torch.sum(KLD_element).mul_(-0.5)

    elif dist == 'cat':
        pi_sum = mu_z.sum(dim=1)
        neg_entropy = (torch.log(mu_z + EPSILON) * mu_z).sum()
        KLD = neg_entropy - np.log(K)

    return LL + KLD


def train(epoch, model, optimizer, train_stream, num_examples, test_stream,
          test_N):
    model.train()
    train_loss = 0
    enumerator = enumerate(train_stream.get_epoch_iterator())
    print(model)
    for batch_idx, (idx, noun, rel, verb) in enumerator:
        optimizer.zero_grad()
        # data = verb
        data = torch.cat([verb, noun], 1)
        mu_x, logvar_x, mu_z, logvar_z = model(data)
        loss = loss_function(mu_x,
                             logvar_x,
                             data,
                             mu_z,
                             logvar_z,
                             model.dist,
                             model.embedding_size)
        loss.backward()
        train_loss += -loss.data[0]
        train_logger.add_scalar_value('loss', float(-loss.data[0]) / len(data))
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # test(epoch, model, test_stream, test_N)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), num_examples, 100. * (
                    batch_idx * len(data)) / num_examples, -loss.data[0] / len(
                        data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_examples))
    # test(epoch, model, test_stream, test_N)


def test(epoch, model, test_stream, num_examples):
    model.eval()
    test_loss = 0
    enumerator = test_stream.get_epoch_iterator()
    for idx, noun, rel, verb in enumerator:
        data = torch.cat([verb, noun], 1)
        # data = verb
        mu_x, logvar_x, mu_z, logvar_z = model(data)
        loss = loss_function(mu_x,
                             logvar_x,
                             data,
                             mu_z,
                             logvar_z,
                             dist=model.dist,
                             K=model.embedding_size).data[0]
        test_loss += loss

    test_loss = -test_loss / num_examples
    test_logger.add_scalar_value(
        'loss', test_loss, step=train_logger.scalar_steps['loss'])
    print('====> Test set loss: {:.4f}'.format(test_loss))


class ToVariable(AgnosticSourcewiseTransformer):
    def __init__(self, datastream, cuda=True, volatile=False, *args, **kwargs):
        super().__init__(
            data_stream=datastream,
            produces_examples=datastream.produces_examples,
            **kwargs)
        self.cuda = cuda
        self.volatile = volatile

    def transform_any_source(self, source, *args, **kwargs):
        source = torch.Tensor(source)
        if self.cuda:
            source = source.cuda()
        return torch.autograd.Variable(source, volatile=self.volatile)


def load_data(use_cuda=True):
    train_data = H5PYDataset(
        '/home/jorn/Desktop/outfile.test', which_sets=('train', ))
    test_data = H5PYDataset(
        '/home/jorn/Desktop/outfile.test', which_sets=('test', ))

    def stream(data, batch_size=128, volatile=False):
        scheme = SequentialScheme(
            examples=data.num_examples, batch_size=batch_size)
        datastream = DataStream(data, iteration_scheme=scheme)
        datastream = ToVariable(
            datastream, which_sources=('noun_vec', 'verb_vec'), cuda=use_cuda)
        return datastream

    return (stream(train_data, batch_size=1024), stream(
        test_data, batch_size=1024, volatile=True),
            train_data.num_examples, test_data.num_examples)


def create_loggers():
    crayonclient = CrayonClient()
    try:
        crayonclient.remove_experiment("train")
    except ValueError:
        pass
    try:
        crayonclient.remove_experiment("test")
    except ValueError:
        pass
    train_logger = crayonclient.create_experiment("train")
    test_logger = crayonclient.create_experiment("test")
    return train_logger, test_logger


def main():
    train_stream, test_stream, train_N, test_N = load_data(use_cuda=args.cuda)

    model = VAE(vector_size=600, dist='norm')
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, train_stream, train_N, test_stream,
              test_N)
        test(epoch, model, test_stream, test_N)
        if args.save_path:
            torch.save(model, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-path', type=str, default="")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_logger, test_logger = create_loggers()

    main()
