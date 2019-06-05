import logging

import torch  # Note: Import torch first!
import torch_baidu_ctc._C as _torch_baidu_ctc

_logger = logging.getLogger(__name__)


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, (
        "gradients only computed for acts - please mark other tensors as "
        "not requiring gradients"
    )


_double_types = [torch.DoubleTensor]
_supported_types = [torch.FloatTensor, torch.DoubleTensor]
if torch.cuda.is_available():
    _double_types += [torch.cuda.DoubleTensor]
    _supported_types += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

_double_types = tuple(_double_types)
_supported_types = tuple(_supported_types)


class _CTC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, acts, labels, acts_lens, labels_lens, blank=0):
        assert isinstance(
            acts, _supported_types
        ), "Unsupported tensor type {!r}".format(acts.type())
        acts = acts.contiguous()
        labels = labels.contiguous()
        acts_lens = acts_lens.contiguous()
        labels_lens = labels_lens.contiguous()

        if isinstance(acts, _double_types):
            _logger.debug("Tensor converted to float in ctc_loss")

        costs, ctx.grads = _torch_baidu_ctc.ctc_loss(
            x=acts.float(),
            y=labels.int(),
            xs=acts_lens.int(),
            ys=labels_lens.int(),
            blank_label=blank,
        )

        # Convert to the same type/device as the input
        costs = costs.to(acts)
        ctx.grads = ctx.grads.to(acts)

        return costs

    @staticmethod
    def backward(ctx, grad_costs):
        # Broadcast-friendly grad_costs, since grad_cost has shape (N,)
        grad_costs = grad_costs.view(1, grad_costs.numel(), 1)
        grad_costs = grad_costs.to(ctx.grads)
        return ctx.grads * grad_costs, None, None, None, None


def ctc_loss(
    acts,  # type: torch.FloatTensor
    labels,  # type: torch.LongTensor
    acts_lens,  # type: torch.LongTensor
    labels_lens,  # type: torch.LongTensor
    average_frames=False,  # type: bool
    reduction=None,  # type: Optional[AnyStr]
    blank=0,  # type: int
):
    """The Connectionist Temporal Classification loss.

    Args:
      acts (torch.Tensor): Input tensor (float or double) with shape
        (T, N, L) where T is the maximum number of output frames, N is the
        minibatch size, and L is the number of output labels (including the
        CTC blank).
      labels (torch.LongTensor): Tensor with shape (K,) representing the
        reference labels for all samples in the minibatch.
      acts_lens (torch.LongTensor): Tensor with shape (N,) representing the
        number of frames for each sample in the minibatch.
      labels_lens (torch.LongTensor): Tensor with shape (N,) representing the
        length of the transcription for each sample in the minibatch.
      average_frames (bool, optional): Specifies whether the loss of each
        sample should be divided by its number of frames. Default: ``False''.
      reduction (string, optional): Specifies the type of reduction to
        perform on the minibatch costs: 'none' | 'mean' | 'sum'.
        With 'none': no reduction is done and a tensor with the cost of each
        sample in the minibatch is returned,
        'mean': the mean of all costs in the minibatch is returned,
        'sum': the sum of all costs in the minibatch is returned.
        Default: 'sum'.
      blank (int, optional): label used to represent the CTC blank symbol.
        Default: 0.
    """
    # type: (...) -> torch.Tensor
    assert average_frames is None or isinstance(average_frames, bool)
    assert reduction is None or reduction in ("none", "mean", "sum")
    assert isinstance(blank, int)

    _assert_no_grad(labels)
    _assert_no_grad(acts_lens)
    _assert_no_grad(labels_lens)

    costs = _CTC.apply(acts, labels, acts_lens, labels_lens, blank)

    if average_frames:
        costs = costs / acts_lens.to(acts)

    if reduction is None:
        reduction = "sum"

    if reduction == "none":
        return costs
    elif reduction == "mean":
        return costs.mean()
    elif reduction == "sum":
        return costs.sum()
    else:
        raise ValueError("Unsupported reduction type {!r}".format(reduction))


class CTCLoss(torch.nn.Module):
    """The Connectionist Temporal Classification loss.

    Args:
      average_frames (bool, optional): Specifies whether the loss of each
        sample should be divided by its number of frames. Default: ``False''.
      reduction (string, optional): Specifies the type of reduction to
        perform on the minibatch costs: 'none' | 'mean' | 'sum'.
        With 'none': no reduction is done and a tensor with the cost of each
        sample in the minibatch is returned,
        'mean': the mean of all costs in the minibatch is returned,
        'sum': the sum of all costs in the minibatch is returned.
        Default: 'sum'.
      blank (int, optional): label used to represent the CTC blank symbol.
        Default: 0.
    """

    def __init__(self, average_frames=None, reduction=None, blank=0):
        assert average_frames is None or isinstance(average_frames, bool)
        assert reduction is None or reduction in ("none", "mean", "sum")
        assert isinstance(blank, int)
        super(CTCLoss, self).__init__()
        self.blank = blank
        self.average_frames = average_frames
        self.reduction = reduction

    def forward(self, acts, labels, acts_lens, labels_lens):
        """CTC forward pass.
        Args:
          acts (torch.Tensor): Input tensor (float or double)  with shape
            (T, N, L), where T is the maximum number of output frames,
            N is the minibatch size, and L is the number of output labels
            (including the CTC blank).
          labels (torch.LongTensor): Tensor with shape (K,) representing the
            reference labels for all samples in the minibatch.
          acts_lens (torch.LongTensor): Tensor with shape (N,) representing the
            number of frames for each sample in the minibatch.
          labels_lens (torch.LongTensor): Tensor with shape (N,) representing
            the length of the transcription for each sample in the minibatch.
        """
        return ctc_loss(
            acts=acts,
            labels=labels,
            acts_lens=acts_lens,
            labels_lens=labels_lens,
            average_frames=self.average_frames,
            reduction=self.reduction,
            blank=self.blank,
        )
