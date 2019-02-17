from __future__ import division

import unittest

import itertools
import numpy as np
import torch
from torch.nn.functional import log_softmax

import torch_baidu_ctc


class CTCLossTest(unittest.TestCase):
    @staticmethod
    def _create_test_data(dtype, device, average_frames, reduction):
        # Size: T x N x 3
        x = torch.tensor(
            [
                [[0, 1, 2], [2, 3, 1], [0, 0, 1]],
                [[-1, -1, 1], [-3, -2, 2], [1, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [1, 1, 1]],
                [[0, 0, 2], [0, 0, -1], [0, 2, 1]],
            ],
            dtype=dtype,
            device=device,
        )
        # Normalize log probabilities
        xn = log_softmax(x, dim=-1)
        xs = torch.tensor([4, 3, 4], dtype=torch.long)
        y = torch.tensor([1, 1, 2, 1, 1, 2, 2], dtype=torch.long)
        ys = torch.tensor([1, 3, 3], dtype=torch.long)
        # Cost of all paths for the reference labeling of each sample
        paths0 = torch.tensor(
            [
                xn[0, 0, a] + xn[1, 0, b] + xn[2, 0, c] + xn[3, 0, d]
                for a, b, c, d in [
                    (1, 1, 1, 1),
                    (1, 1, 1, 0),
                    (0, 1, 1, 1),
                    (1, 1, 0, 0),
                    (0, 1, 1, 0),
                    (0, 0, 1, 1),
                    (1, 0, 0, 0),
                    (0, 1, 0, 0),
                    (0, 0, 1, 0),
                    (0, 0, 0, 1),
                ]
            ],
            device=device,
        )
        paths1 = xn[0, 1, 1] + xn[1, 1, 2] + xn[2, 1, 1]
        paths2 = xn[0, 2, 1] + xn[1, 2, 2] + xn[2, 2, 0] + xn[3, 2, 2]
        expected = torch.cat(
            [-torch.logsumexp(paths0, 0).view(1), -paths1.view(1), -paths2.view(1)]
        ).cpu()
        if average_frames:
            expected = expected / xs.type(dtype)

        if reduction == "sum":
            expected = torch.sum(expected)
        elif reduction == "mean":
            expected = torch.mean(expected)

        return x, y, xs, ys, expected

    @staticmethod
    def _run_test_forward(dtype, device, average_frames, reduction):
        x, y, xs, ys, expected = CTCLossTest._create_test_data(
            dtype, device, average_frames, reduction
        )
        # Test function
        loss = torch_baidu_ctc.ctc_loss(
            x, y, xs, ys, average_frames=average_frames, reduction=reduction
        )
        np.testing.assert_array_almost_equal(loss.cpu(), expected.cpu())

        # Test module
        loss = torch_baidu_ctc.CTCLoss(
            average_frames=average_frames, reduction=reduction
        )(x, y, xs, ys)
        np.testing.assert_array_almost_equal(loss.cpu(), expected.cpu())

    @staticmethod
    def _run_test_grad(dtype, device, average_frames, reduction):
        problem_sizes = [(20, 50, 15, 1, 10 ** (-2.5)), (5, 10, 5, 65, 1e-2)]
        for alphabet_size, num_frames, labels_len, minibatch, tol in problem_sizes:
            x = torch.rand(
                num_frames,
                minibatch,
                alphabet_size,
                requires_grad=True,
                dtype=dtype,
                device=device,
            )
            y = torch.zeros(labels_len, minibatch, dtype=torch.int).random_(
                1, alphabet_size
            )
            xs = torch.zeros(minibatch, dtype=torch.int).fill_(num_frames)
            ys = torch.zeros(minibatch, dtype=torch.int).fill_(labels_len)
            if labels_len >= 3:  # guarantee repeats for testing
                y[labels_len // 2] = y[labels_len // 2 + 1]
                y[labels_len // 2 - 1] = y[labels_len // 2]
            y = y.view(-1)

            def f_(x_):
                loss = torch_baidu_ctc.ctc_loss(
                    x_, y, xs, ys, average_frames=average_frames, reduction=reduction
                )
                return torch.sum(loss / 2.0)

            torch.autograd.gradcheck(f_, (x,), rtol=tol, atol=tol, eps=1e-1)


def _generate_test(test_name, method, dtype, device, average_frames, reduction):
    avg_str = "avg" if average_frames else "no_avg"
    setattr(
        CTCLossTest,
        test_name + "_{}_{}_{}_{}".format(avg_str, reduction, device, str(dtype)[6:]),
        lambda self: getattr(self, method)(dtype, device, average_frames, reduction),
    )


devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
dtypes = [torch.float32, torch.float64]
average_frames = [False, True]
reductions = ["none", "mean", "sum"]

for dtype, device, avg, reduction in itertools.product(
    dtypes, devices, average_frames, reductions
):
    _generate_test("test_forward", "_run_test_forward", dtype, device, avg, reduction)
    _generate_test("test_grad", "_run_test_grad", dtype, device, avg, reduction)


if __name__ == "__main__":
    unittest.main()
