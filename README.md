# SlicerPyTorch

<p align="center">
  <img src="https://raw.githubusercontent.com/fepegar/SlicerPyTorch/master/PyTorch.png" alt="SlicerPyTorch logo" width=300>
</p>

This is the code for the `PyTorch` extension for [3D Slicer](https://www.slicer.org/).

Its main function is to install [PyTorch](https://pytorch.org/) inside Slicer. The latest version compatible with the installed drivers will be selected automatically.

PyTorch can be installed opening the `PyTorch Utils` module and clicking on the button, or programmatically:

```python
import PyTorchUtils
torch = PyTorchUtils.PyTorchUtilsLogic().torch  # will be installed if necessary
tensor = torch.rand(50, 60)
```

Here's a diagram of the integration between PyTorch and Slicer:

![Diagram of PyTorch and Slicer](https://raw.githubusercontent.com/NA-MIC/ProjectWeek/master/PW35_2021_Virtual/Projects/PyTorchIntegration/diagram.svg)
