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


Unit tests are run nightly and the results can be checked on [CDash](https://slicer.cdash.org/index.php?project=SlicerPreview&filtercount=1&showfilters=1&field1=buildname&compare1=63&value1=PyTorch).

Here's a diagram of the integration between PyTorch and Slicer:

![Diagram of PyTorch and Slicer](https://raw.githubusercontent.com/NA-MIC/ProjectWeek/master/PW35_2021_Virtual/Projects/PyTorchIntegration/diagram.svg)
