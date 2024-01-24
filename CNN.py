# --coding:utf-8--
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math

n_epochs = 20
batch_size_train = 64
batch_size_test = 100
learning_rate = 0.01
momentum = 0.5
log_interval = 400
random_seed = 1
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def clipping(grad, C):
    # OrgGradient must be a vector,C is the clipping theoreshold
    L2 = torch.norm(grad) / C
    if L2 > 1:
        return grad / L2
    else:
        return grad


def cartesian_to_hyperspherical(coords):
    n = coords.shape[0]
    angles = torch.zeros(n - 1)
    radius = torch.norm(coords)

    for i in range(n - 2):
        angles[i] = torch.acos(coords[i] / torch.norm(coords[i:n]))

    angles[n - 2] = 2 * (math.pi / 2 - torch.atan((coords[n - 2] + torch.norm(coords[n - 2:])) / coords[n - 1]))

    return radius, angles


def hyperspherical_to_cartesian(radius, angles):
    n = angles.shape[0] + 1
    coords = torch.zeros(n)

    coords[0] = radius * torch.cos(angles[0])

    for i in range(1, n - 1):
        coords[i] = radius * torch.prod(torch.sin(angles[:i])) * torch.cos(angles[i])

    coords[n - 1] = radius * torch.prod(torch.sin(angles))

    return coords

def NumPer(grad,sigma,C,B):
    clipped_grad = clipping(grad, C)
    r = clipped_grad.shape[0]
    return clipped_grad + C * (torch.randn(r) * sigma)/B

def VecPer(grad,sigma,C,B):
    beta=0.1
    clipped_grad = clipping(grad, C)
    r = clipped_grad.shape[0]
    radius, angles = cartesian_to_hyperspherical(clipped_grad)
    radius_with_noise = radius + C * (torch.randn(1) * sigma)/B
    angles_with_noise = torch.zeros(r - 1)
    C_Ang=math.sqrt(r+2)*beta*math.pi
    angles_with_noise = angles + C_Ang * (torch.randn(r - 1) * sigma)
    noised_grad = hyperspherical_to_cartesian(radius_with_noise, angles_with_noise)
    return noised_grad

def add_noise_by_scalar(model, sigma, C, B):
    # get grads and flatten
    grad_shape_list = []
    flatten_grad = torch.tensor([1])
    for params in model.parameters():
        grad = params.grad
        grad_shape_list.append(torch.prod(torch.tensor(grad.shape)))
        grad_vector = torch.reshape(grad, (-1,))
        flatten_grad = torch.cat((flatten_grad, grad_vector))
    flatten_grad = flatten_grad[1:]
    grad_shape_tensor = torch.tensor(grad_shape_list)

    # add noise
    grad_with_noise = NumPer(flatten_grad, sigma, C,B)

    # change the original grads
    start_point = 0
    for index, params in enumerate(model.parameters()):
        end_point = torch.sum(grad_shape_tensor[:index + 1])
        params.grad.data = torch.reshape(grad_with_noise[start_point:end_point],
                                         tuple(params.grad.shape))
        start_point = end_point


# add noise by vector
def add_noise_by_vector(model, sigma, C,  B):
    # get grads and flatten
    grad_shape_list = []
    flatten_grad = torch.tensor([1])
    for params in model.parameters():
        grad = params.grad
        grad_shape_list.append(torch.prod(torch.tensor(grad.shape)))
        grad_vector = torch.reshape(grad, (-1,))
        flatten_grad = torch.cat((flatten_grad, grad_vector))
    flatten_grad = flatten_grad[1:]
    grad_shape_tensor = torch.tensor(grad_shape_list)

    # add noise
    grad_with_noise = VecPer(flatten_grad, sigma, C, B)

    # change the original grads
    start_point = 0
    for index, params in enumerate(model.parameters()):
        end_point = torch.sum(grad_shape_tensor[:index + 1])
        params.grad.data = torch.reshape(grad_with_noise[start_point:end_point],
                                         tuple(params.grad.shape))
        start_point = end_point


def train(epoch, network, optimizer, noising_idx, sigma, C, B):
    train_losses = []
    train_counter = []
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        if noising_idx == 'num':
            add_noise_by_scalar(network, sigma, C, B)
        elif noising_idx == 'vec':
            add_noise_by_vector(network, sigma, C, B)

        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), 'savings/model_{}_{}.pth'.format(noising_idx, epoch))
            torch.save(optimizer.state_dict(), 'savings/optimizer_{}_{}.pth'.format(noising_idx, epoch))
    return train_losses, train_counter


def test(network):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * accuracy))
    return test_loss, accuracy


def run_models(noising_idx, sigma, B, C):
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    accuracy = []
    for epoch in range(1, n_epochs + 1):
        train_loss_per_epoch, train_counter_per_epoch = train(epoch, network, optimizer, noising_idx, sigma, C, B)
        test_loss_per_epoch, accu = test(network)

        train_losses.extend(train_loss_per_epoch)
        train_counter.extend(train_counter_per_epoch)

        test_losses.append(test_loss_per_epoch)
        accuracy.append(accu)

    return train_losses, train_counter, test_losses, accuracy


if __name__ == '__main__':
    # examples = enumerate(test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
    # print(example_targets)
    # print(example_data.shape)

    # fig = plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #     plt.title("Ground Truth: {}".format(example_targets[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()
    sigma=10
    C = 0.1
    B = batch_size_train
    test_counter = [i for i in range(1, n_epochs + 1)]

    train_losses_num, train_counter_num, test_losses_num, accuracy_num = run_models('num', sigma, C, B)

    train_losses_vec, train_counter_vec, test_losses_vec, accuracy_vec = run_models('vec', sigma, C, B)

    train_losses_none, train_counter_none, test_losses_none, accuracy_none = run_models('none', sigma, C, B)

    fig = plt.figure()
    plt.plot(train_counter_none, train_losses_none, color='red')
    plt.plot(train_counter_num, train_losses_num, color='green')
    plt.plot(train_counter_vec, train_losses_vec, color='blue')
    plt.legend(['None-Noise-Grad', 'Noise-by-Num-Grad', 'Noise-by-Vec-Grad'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood training loss')
    # plt.title('Grad Noise by NUM')
    plt.savefig('training_loss.tiff')

    fig = plt.figure()
    plt.plot(test_counter, test_losses_none, color='red')
    plt.plot(test_counter, test_losses_num, color='green')
    plt.plot(test_counter, test_losses_vec, color='blue')
    plt.legend(['None-Noise-Grad', 'Noise-by-Num-Grad', 'Noise-by-Vec-Grad'], loc='upper right')
    plt.xlabel('number of epoches seen')
    plt.ylabel('negative log likelihood testing loss')
    plt.savefig('testing_loss.tiff')

    fig = plt.figure()
    plt.plot(test_counter, accuracy_none, color='red')
    plt.plot(test_counter, accuracy_num, color='green')
    plt.plot(test_counter, accuracy_vec, color='blue')
    plt.legend(['None-Noise-Grad', 'Noise-by-Num-Grad', 'Noise-by-Vec-Grad'], loc='lower right')
    plt.xlabel('number of epoches seen')
    plt.ylabel('accuracy')
    plt.savefig('accuracy.tiff')

    # train_losses_vec, train_counter_vec, test_losses_vec, accuracy_vec = run_models(noising_idx='vec', sigma=sigma, C=C)

    data_dict = {'vec': [train_losses_vec, train_counter_vec, test_losses_vec, accuracy_vec],
                 'num': [train_losses_num, train_counter_num, test_losses_num, accuracy_num],
                 'none': [train_losses_none, train_counter_none, test_losses_none, accuracy_none]}

    torch.save(data_dict, 'results.pkl')

    # examples = enumerate(test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
    # with torch.no_grad():
    #     output = network(example_data)
    # fig = plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #     plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
    #     plt.xticks([])
    #     plt.yticks([])

    # ----------------------------------------------------------- #
    #
    # continued_network = Net()
    # continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    #
    # network_state_dict = torch.load('model.pth')
    # continued_network.load_state_dict(network_state_dict)
    # optimizer_state_dict = torch.load('optimizer.pth')
    # continued_optimizer.load_state_dict(optimizer_state_dict)
    #
    # # 注意不要注释前面的“for epoch in range(1, n_epochs + 1):”部分，
    # # 不然报错：x and y must be the same size
    # # 为什么是“4”开始呢，因为n_epochs=3，上面用了[1, n_epochs + 1)
    # for i in range(4, 9):
    #     test_counter.append(i * len(train_loader.dataset))
    #     train(i)
    #     test()
    #
    # fig = plt.figure()
    # plt.plot(train_counter, train_losses, color='blue')
    # plt.scatter(test_counter, test_losses, color='red')
    # plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    # plt.xlabel('number of training examples seen')
    # plt.ylabel('negative log likelihood loss')
    # plt.show()
