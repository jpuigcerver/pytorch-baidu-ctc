#include <cstring>

#include <tuple>
#include <sstream>
#include <string>

#include <torch/torch.h>
#include <ctc.h>

#include <ATen/Context.h>
#include <ATen/CPUGeneral.h>
#include <ATen/Device.h>
#include <ATen/DeviceGuard.h>


#define CHECK_CONTIGUOUS(x)                                       \
  AT_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CHECK_CPU(x)                                            \
  AT_CHECK((x).device().type() == at::Device::Type::CPU,        \
           #x " must be located in the CPU")

#define CHECK_CPU_OR_CUDA(x)                                    \
  AT_CHECK(((x).device().type() == at::Device::Type::CPU ||     \
            (x).device().type() == at::Device::Type::CUDA),     \
           #x " must be located in the CPU or a CUDA device")

#define CHECK_FLOAT(x)                                                  \
  AT_CHECK((x).type().scalarType() == at::ScalarType::Float,            \
           #x " must be a Float tensor")

#define CHECK_INT(x)                                                    \
  AT_CHECK((x).type().scalarType() == at::ScalarType::Int,              \
           #x " must be a Int tensor")

#define CHECK_NUM_DIM_IS_2_OR_3(x)              \
  AT_CHECK((x).dim() == 2 || (x).dim() == 3,    \
           #x " must have 2 or 3 dimensions")

#define CHECK_SAME_NUM_ELEMENTS(t1, t2) do {                                 \
    const auto s1 = (t1).numel();                                            \
    const auto s2 = (t2).numel();                                            \
    AT_CHECK(s1 == s2,                                                       \
             "Number of elements of " #t1 " and " #t2 " must be equal "      \
             "(" + std::to_string(s1) + " vs. " + std::to_string(s2) + ")"); \
  } while(0)

#define CHECK_WARP_CTC_CALL(s) do {                             \
    const ctcStatus_t status = (s);                             \
    const std::string status_str(ctcGetStatusString(status));   \
    AT_CHECK(status == CTC_STATUS_SUCCESS,                      \
             "ctc_loss failed with status " + status_str);      \
  } while(0)


std::tuple<at::Tensor, at::Tensor> ctc_loss(
    const at::Tensor& x, const at::Tensor& y,
    const at::Tensor& xs, const at::Tensor& ys,
    const int blank_label) {
  // Check contiguous
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(y);
  CHECK_CONTIGUOUS(xs);
  CHECK_CONTIGUOUS(ys);
  // Check types
  CHECK_FLOAT(x);
  CHECK_INT(y);
  CHECK_INT(xs);
  CHECK_INT(ys);
  // Check device
  CHECK_CPU_OR_CUDA(x);
  CHECK_CPU(y);
  CHECK_CPU(xs);
  CHECK_CPU(ys);
  // Check number of dimensions and elements
  CHECK_NUM_DIM_IS_2_OR_3(x);
  CHECK_SAME_NUM_ELEMENTS(xs, ys);

  const auto minibatch = xs.numel();
  const auto alphabet_size = x.dim() == 2 ? x.size(1) : x.size(2);

  // Allocate memory for input gradient
  at::Tensor grad_x = at::zeros_like(x);
  // Allocate memory for losses
  at::Tensor losses = at::zeros_like(ys).to(at::ScalarType::Float);

  ctcOptions ctc_opts;
  memset(&ctc_opts, 0, sizeof(ctcOptions));
  ctc_opts.blank_label = blank_label;
  if (x.device().type() == at::Device::Type::CPU) {
    ctc_opts.loc = CTC_CPU;
    ctc_opts.num_threads = std::max<unsigned int>(at::get_num_threads(), 0);
  } else {
    ctc_opts.loc = CTC_GPU;
    const auto index = x.device().index();
    ctc_opts.stream =
        at::globalContext().getCurrentCUDAStreamOnDevice(index).stream();
  }

  // Allocate workspace memory
  size_t workspace_size = 0;
  CHECK_WARP_CTC_CALL(
      get_workspace_size(
          ys.data<int>(), xs.data<int>(), alphabet_size, minibatch, ctc_opts,
          &workspace_size));

  at::TensorOptions workspace_opts(x.device());
  workspace_opts.dtype(at::ScalarType::Byte);
  at::Tensor workspace =
      at::zeros({static_cast<int64_t>(workspace_size * 10)}, workspace_opts);

  at::DeviceGuard device_guard(x.device());
  CHECK_WARP_CTC_CALL(
      compute_ctc_loss(
          x.data<float>(),
          grad_x.data<float>(),
          y.data<int>(),
          ys.data<int>(),
          xs.data<int>(),
          alphabet_size,
          minibatch,
          losses.data<float>(),
          workspace.data<uint8_t>(),
          ctc_opts));

  return std::make_tuple(losses, grad_x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ctc_loss",
        &ctc_loss,
        "Baidu's CTC loss (forward and backward).",
        pybind11::arg("x"),
        pybind11::arg("y"),
        pybind11::arg("xs"),
        pybind11::arg("ys"),
        pybind11::arg("blank_label") = 0);
}
