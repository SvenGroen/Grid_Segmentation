import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from code.utils.SegLoss import *
import code.utils.SegLoss


# https://www.jeremyjordan.me/semantic-segmentation/#loss

def get_weights(ground_truth):
    total = torch.prod(torch.tensor(ground_truth.size())).float()
    bg = (ground_truth == 0).sum() / total
    return torch.tensor([bg.item(), (1 - bg).item()])





def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    y_true = torch.transpose(y_true, 1, -1).numpy()
    y_pred = torch.transpose(y_pred, 1, -1).numpy()
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    return 1 - np.mean((numerator + epsilon) / (denominator + epsilon))  # average over classes and batch

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target

if __name__ == '__main__':
    true = torch.tensor([[
        [
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.]
        ]]])
    pred = torch.tensor([
        [
            [
                [1., 1., 1.],
                [1., 1., 0.],
                [1., 1., 1.]
            ],
            [
                [0., 0., 0.],
                [0., 1., 0.],
                [0., 0., 0.]
            ]
        ]
    ])
    print(true.shape, pred.shape)
    a = make_one_hot(true.long(), C=2)
    print(soft_dice_loss(a, pred, epsilon=0.0001))

    # print(get_weights(true))
    # loss = F.cross_entropy(pred, true.long(), weight=torch.tensor([0.,1.]))
    # pred2 = pred.argmax(dim=1)
    # print(pred2)
    # loss2 = F.binary_cross_entropy(pred2.float(), true.float(), weight=torch.tensor([1.]))
    # print(loss, loss2)
    floss = focal_loss.FocalLoss()
    dloss = dice_loss.SoftDiceLoss(smooth=0.0001)
    bloss = boundary_loss.BDLoss()
    mask = true.squeeze(0).numpy()
    mask = torch.tensor(np.expand_dims(mask, axis=1))
    #
    #
    #
    # print(floss(pred, true))
    # print(1- -1*dloss(pred, true))
    """
    for boundary loss
            net_output: (batch_size, class, x,y,z)
            target: ground truth, shape: (batch_size, 1, x,y,z)
            bound: precomputed distance map, shape (batch_size, class, x,y,z)
    """

    print(pred.shape,true.shape,  a.shape)
    print(bloss(pred, true, a))
