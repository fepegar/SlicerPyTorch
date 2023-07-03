import qt
import logging

import slicer
from slicer.ScriptedLoadableModule import (
  ScriptedLoadableModule,
  ScriptedLoadableModuleWidget,
  ScriptedLoadableModuleLogic,
  ScriptedLoadableModuleTest,
)


class PyTorchUtils(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "PyTorch Utils"
    self.parent.categories = ['Utilities']
    self.parent.dependencies = []
    self.parent.contributors = [
      "Fernando Perez-Garcia (University College London and King's College London)",
      "Andras Lasso (PerkLab Queen's University)",
    ]
    self.parent.helpText = 'This hidden module containing some tools to work with PyTorch inside Slicer.'
    self.parent.acknowledgementText = (
      'This work was funded by the Engineering and Physical Sciences'
      ' Research Council (EPSRC) and supported by the UCL Centre for Doctoral'
      ' Training in Intelligent, Integrated Imaging in Healthcare, the UCL'
      ' Wellcome / EPSRC Centre for Interventional and Surgical Sciences (WEISS),'
      ' and the School of Biomedical Engineering & Imaging Sciences (BMEIS)'
      " of King's College London."
    )


class PyTorchUtilsWidget(ScriptedLoadableModuleWidget):

  def setup(self):
    super().setup()

    self.logic = PyTorchUtilsLogic()

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/PyTorchUtils.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    self.ui.detectPushButton.clicked.connect(self.onDetect)
    self.ui.installPushButton.clicked.connect(self.onInstallTorch)
    self.ui.uninstallPushButton.clicked.connect(self.onUninstallTorch)
    self.ui.restartPushButton.clicked.connect(self.onApplicationRestart)

    self.updateVersionInformation()

  def onDetect(self):
    with slicer.util.tryWithErrorDisplay("Failed to detect compatible computation backends.", waitCursor=True):
      torchVersionRequirement = self.ui.torchVersionLineEdit.text
      backends = PyTorchUtilsLogic.getCompatibleComputationBackends(torchVersionRequirement=torchVersionRequirement)
      currentBackend = self.ui.backendComboBox.currentText
      self.ui.backendComboBox.clear()
      self.ui.backendComboBox.addItem("automatic")
      for backend in backends:
        self.ui.backendComboBox.addItem(backend)
      self.ui.backendComboBox.currentText = currentBackend
      self.ui.backendComboBox.showPopup()
    self.updateVersionInformation()

  def onInstallTorch(self):
    with slicer.util.tryWithErrorDisplay("Failed to install PyTorch. Some PyTorch files may be in use or corrupted. Please restart the application, uninstall PyTorch, and try installing again.", waitCursor=True):
      if PyTorchUtilsLogic.torchInstalled():
        torch = self.logic.torch
        slicer.util.delayDisplay(f'PyTorch {torch.__version__} is already installed, using {self.logic.getDevice()}.', autoCloseMsec=2000)
      else:
        backend = self.ui.backendComboBox.currentText
        automaticBackend = (backend == "automatic")
        askConfirmation = automaticBackend
        torchVersionRequirement = self.ui.torchVersionLineEdit.text
        torchvisionVersionRequirement = self.ui.torchvisionVersionLineEdit.text
        torch = self.logic.installTorch(askConfirmation, None if automaticBackend else backend, torchVersionRequirement, torchvisionVersionRequirement)
        if torch is not None:
          slicer.util.delayDisplay(f'PyTorch {torch.__version__} installed successfully using {self.logic.getDevice()}.', autoCloseMsec=2000)
    self.updateVersionInformation()

  def onUninstallTorch(self):
    with slicer.util.tryWithErrorDisplay("Failed to uninstall PyTorch. Probably PyTorch is already in use. Please restart the application and try again.", waitCursor=True):
      self.logic.uninstallTorch()
      slicer.util.delayDisplay(f'PyTorch uninstalled successfully.', autoCloseMsec=2000)
    self.updateVersionInformation()

  def updateVersionInformation(self):
    try:
      self.ui.torchVersionInformation.text = self.logic.torchVersionInformation
    except Exception as e:
      logging.error(str(e))
      self.ui.torchVersionInformation.text = "unknown (corrupted installation?)"
    try:
      self.ui.torchvisionVersionInformation.text = self.logic.torchvisionVersionInformation
    except Exception as e:
      logging.error(str(e))
      self.ui.torchvisionVersionInformation.text = "unknown (corrupted installation?)"
    try:
      info = self.logic.nvidiaDriverVersionInformation
      self.ui.nvidiaVersionInformation.text = info if info else "not found"
    except Exception as e:
      logging.error(str(e))
      self.ui.nvidiaVersionInformation.text = "not found"

  def onApplicationRestart(self):
    slicer.util.restart()

class PyTorchUtilsLogic(ScriptedLoadableModuleLogic):
  def __init__(self):
    self._torch = None

  @property
  def nvidiaDriverVersionInformation(self):
    """Get NVIDIA driver version information as a string that can be displayed to the user.
    If light-the-torch is not installed yet then empty string is returned.
    """

    try:
      import light_the_torch._cb as computationBackend
      version = computationBackend._detect_nvidia_driver_version()
      if version is None:
        return ""
      else:
        return f"installed version {str(version)}"
    except Exception as e:
      # Don't install light-the-torch just for getting the NVIDIA driver version
      return ""

  @property
  def torchVersionInformation(self):
    """Get PyTorch version information as a string that can be displayed to the user.
    """
    if not PyTorchUtilsLogic.torchInstalled():
      return "not installed"
    import torch
    return f"installed version {torch.__version__}"

  @property
  def torchvisionVersionInformation(self):
    """Get TorchVision version information as a string that can be displayed to the user.
    """
    if not PyTorchUtilsLogic.torchvisionInstalled():
      return "not installed"
    import torchvision
    return f"installed version {torchvision.__version__}"

  @property
  def torch(self):
    """``torch`` Python module. it will be installed if necessary."""
    if self._torch is None:
      logging.info('Importing torch...')
      self._torch = self.importTorch()
    return self._torch

  @staticmethod
  def torchInstalled():
    # Attempt to import torch could load some files, which could prevent uninstalling a corrupted pytorch install
    import importlib.metadata
    try:
      metadataPath = [p for p in importlib.metadata.files('torch') if 'METADATA' in str(p)][0]
    except importlib.metadata.PackageNotFoundError as e:
      return False
    try:
      import torch
      installed = True
    except ModuleNotFoundError:
      installed = False
    return installed

  @staticmethod
  def torchvisionInstalled():
    try:
      import torchvision
      installed = True
    except ModuleNotFoundError:
      installed = False
    return installed

  def importTorch(self):
    """Import the ``torch`` Python module, installing it if necessary."""
    if PyTorchUtilsLogic.torchInstalled():
      import torch
    else:
      torch = self.installTorch()
    if torch is None:
      logging.warning('PyTorch was not installed')
    else:
      logging.info(f'PyTorch {torch.__version__} imported successfully')
      logging.info(f'CUDA available: {torch.cuda.is_available()}')
    return torch

  def installTorch(self, askConfirmation=False, forceComputationBackend=None, torchVersionRequirement=None, torchvisionVersionRequirement=None):
    """Install PyTorch and return the ``torch`` Python module.

    :param forceComputationBackend: optional parameter to set computation backend (cpu, cu116, cu117, ...)
    :param torchVersionRequirement: optional version requirement for torch (e.g., ">=1.12")
    :param torchvisionVersionRequirement: optional version requirement for torchvision (e.g., ">=0.8")

    If computation backend is not specified then the ``light-the-torch`` Python package is used to get the most recent version of
    PyTorch compatible with the installed NVIDIA drivers. If CUDA-compatible device is not found, a version compiled for CPU will be installed.
    """

    args = PyTorchUtilsLogic._getPipInstallArguments(forceComputationBackend, torchVersionRequirement, torchvisionVersionRequirement)

    if askConfirmation and not slicer.app.commandOptions().testingEnabled:
      install = slicer.util.confirmOkCancelDisplay(
        f'PyTorch will be downloaded and installed using light-the-torch (ltt {" ".join(args)}).'
        ' The process might take some minutes.'
      )
      if not install:
        logging.info('Installation of PyTorch aborted by user')
        return None

    logging.info(f"Install PyTorch using light-the-torch with arguments: {args}")

    try:
      import light_the_torch._patch
    except:
      PyTorchUtilsLogic._installLightTheTorch()
      import light_the_torch._patch

    slicer.util._executePythonModule('light_the_torch', args)
    import torch
    logging.info(f'PyTorch {torch.__version__} installed successfully.')
    return torch

  def uninstallTorch(self, askConfirmation=False, forceComputationBackend=None):
    """Uninstall PyTorch"""
    slicer.util.pip_uninstall('torch torchvision')
    logging.info(f'PyTorch uninstalled successfully.')

  @staticmethod
  def _getPipInstallArguments(forceComputationBackend=None, torchVersionRequirement=None, torchvisionVersionRequirement=None):
    if torchVersionRequirement is None:
      torchVersionRequirement = ""
    if torchvisionVersionRequirement is None:
      torchvisionVersionRequirement = ""
    args = ["install", "torch"+torchVersionRequirement, "torchvision"+torchvisionVersionRequirement]
    if forceComputationBackend is not None:
      args.append(f"--pytorch-computation-backend={forceComputationBackend}")
    return args

  @staticmethod
  def _installLightTheTorch():
    slicer.util.pip_install('light-the-torch>=0.5')

  @staticmethod
  def getCompatibleComputationBackends(forceComputationBackend=None, torchVersionRequirement=None):
    """Get the list of computation backends compatible with the available hardware.

    :param forceComputationBackend: optional parameter to set computation backend (cpu, cu116, cu117, ...)
    :param torchVersionRequirement: optional version requirement for torch (e.g., ">=1.12")

    If computation backend is not specified then the ``light-the-torch`` is used to get the most recent version of
    PyTorch compatible with the installed NVIDIA drivers.
    """
    try:
      import light_the_torch._patch
    except:
      PyTorchUtilsLogic._installLightTheTorch()
      import light_the_torch._patch

    args = PyTorchUtilsLogic._getPipInstallArguments(forceComputationBackend, torchVersionRequirement)
    try:
      backends = sorted(light_the_torch._patch.LttOptions.from_pip_argv(args).computation_backends)
    except Exception as e:
      logging.warning(str(e))
      raise ValueError(f"Failed to get computation backend. Requested computation backend: `{forceComputationBackend}`.")

    return backends

  def getPyTorchHubModel(self, repoOwner, repoName, modelName, addPretrainedKwarg=True, *args, **kwargs):
    """Use PyTorch Hub to download a PyTorch model, typically pre-trained.

    More information can be found at https://pytorch.org/hub/.
    """
    repo = f'{repoOwner}/{repoName}'
    if addPretrainedKwarg:
      kwargs['pretrained'] = True
    model = self.torch.hub.load(repo, modelName, *args, **kwargs)
    return model

  def getDevice(self):
    """Get CUDA device if available and CPU otherwise."""
    return self.torch.device('cuda') if self.torch.cuda.is_available() else 'cpu'

  @property
  def cuda(self):
    """Return True if a CUDA-compatible device is available."""
    return self.getDevice() != 'cpu'


class PyTorchUtilsTest(ScriptedLoadableModuleTest):

  def runTest(self):
    self.test_PyTorchUtils()

  def _delayDisplay(self, message):
    if not slicer.app.testingEnabled():
      self.delayDisplay(message)

  def test_PyTorchUtils(self):
    self._delayDisplay('Starting the test')
    logic = PyTorchUtilsLogic()
    self._delayDisplay(f'CUDA available: {logic.torch.cuda.is_available()}')
    self._delayDisplay('Test passed!')
