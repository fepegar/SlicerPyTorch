# SlicerPyTorch

<p align="center">
  <img src="https://raw.githubusercontent.com/fepegar/SlicerPyTorch/master/PyTorch.png" alt="SlicerPyTorch logo" width=300>
</p>

This is the code for the `PyTorch` extension for [3D Slicer](https://www.slicer.org/).

Its main function is to install [PyTorch](https://pytorch.org/) inside Slicer.
The latest version compatible with the installed drivers will be selected automatically.
[CI tests](https://slicer.cdash.org/index.php?project=SlicerPreview&filtercount=1&showfilters=1&field1=buildname&compare1=63&value1=PyTorch) are run nightly.

PyTorch can be installed opening the `PyTorch Utils` module and clicking on the button, or programmatically:

```python
import PyTorchUtils
torch = PyTorchUtils.PyTorchUtilsLogic().torch  # will be installed if necessary
tensor = torch.rand(50, 60)
```

## GPU acceleration

CUDA is used when a CUDA-compatible GPU is available. If there is no CUDA GPU but a Vulkan driver is accessible, then
[torchvulkan](https://github.com/zainsharief/torchvulkan) is installed automatically, which enables GPU computation on
AMD, Intel, and Apple Silicon GPUs:

```python
import PyTorchUtils
logic = PyTorchUtils.PyTorchUtilsLogic()
device = logic.getDevice()  # cuda, vulkan, or cpu
model.to(device)
```

torchvulkan is experimental, therefore it is only used when it is actually needed: when no CUDA device is found but a
Vulkan driver is. When PyTorch is installed in that situation, the PyTorch version required by torchvulkan is selected
automatically, so GPU acceleration works without any manual version selection. CUDA is always preferred when a
CUDA-compatible GPU is present.

PyTorch is never modified without asking the user: torchvulkan is installed with a `pip` constraints file that pins the
installed PyTorch packages, so it is installed only if it is compatible with the installed PyTorch version. If an
incompatible PyTorch is already installed, then the extension offers to install the version required by torchvulkan (the
application has to be restarted afterwards, because the already imported PyTorch module cannot be reloaded). If no
torchvulkan wheel is available for the platform (for example, on Intel Macs, or on systems with a glibc older than 2.34)
or the installed PyTorch version is not compatible with it, then computation continues on the CPU (an unusable
installation is removed). Note that Slicer's Python accepts only wheels built for glibc 2.28 or older, while the
Linux torchvulkan wheels require glibc 2.34, so the matching wheel is unpacked directly into the `site-packages`
directory when the system glibc is new enough. Uninstalling PyTorch removes torchvulkan as well.


Unit tests are run nightly and the results can be checked on [CDash](https://slicer.cdash.org/index.php?project=SlicerPreview&filtercount=1&showfilters=1&field1=buildname&compare1=63&value1=PyTorch).

Here's a diagram of the integration between PyTorch and Slicer:

![Diagram of PyTorch and Slicer](https://raw.githubusercontent.com/NA-MIC/ProjectWeek/master/PW35_2021_Virtual/Projects/PyTorchIntegration/diagram.svg)
