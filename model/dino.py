
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DiNo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, netF, netB, netC, args, mlp_layers=2, out_dim=8192, use_bn=True, norm_last_layer=True):

        """
        dim: feature dimension (default: 128)
        out_dim: output dimension (default: 65536)
        """
        super(DiNo, self).__init__()

        self.netF = netF
        self.netB = netB
        self.netC = netC

        self.mlp_layers = mlp_layers
        self.hidden_dim = args.hidden_dim
        self.bottleneck_dim = args.bottleneck_dim
        self.out_dim = out_dim

        # create the encoders
        teacher_mlp, teacher_out = self.dino_head(use_bn=use_bn, norm_last_layer=False)
        self.teacher = nn.Sequential(copy.deepcopy(self.netF), teacher_mlp, teacher_out)
        student_mlp, student_out = self.dino_head(use_bn=use_bn, norm_last_layer=norm_last_layer)
        self.student = nn.Sequential(self.netF, student_mlp, student_out)

        # teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False


    def dino_head(self, use_bn, norm_last_layer):
        mlp_dim = self.netF.in_features
        mlp_layers = max(self.mlp_layers, 1)
        if mlp_layers == 1:
            mlp = nn.Linear(mlp_dim, self.bottleneck_dim)
        else:
            layers = [nn.Linear(mlp_dim, self.hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.GELU())
            for _ in range(mlp_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(self.hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(self.hidden_dim, self.bottleneck_dim))
            mlp = nn.Sequential(*layers)

        mlp.apply(self._init_weights)

        last_layer = nn.utils.weight_norm(nn.Linear(self.bottleneck_dim, self.out_dim, bias=False))
        last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            last_layer.weight_g.requires_grad = False

        return mlp, last_layer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def dino_forward(self, x, is_teacher=True):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            if is_teacher:
                _out = self.teacher(torch.cat(x[start_idx: end_idx]))
            else:
                _out = self.student(torch.cat(x[start_idx: end_idx]))
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx

        return output

    def forward(self, img):
        out = self.netB(self.netF(img))
        out = self.netC(out)
        return out

    def ema_update_teacher(self, m):
        # EMA update for the teacher
        with torch.no_grad():
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, ncrops_t=1, warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_teacher_temp_epochs=0, nepochs=20, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.ncrops_t = ncrops_t
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.ncrops_t)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)