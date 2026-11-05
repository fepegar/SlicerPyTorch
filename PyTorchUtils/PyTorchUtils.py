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
    try:
      self.ui.vulkanVersionInformation.text = self.logic.vulkanVersionInformation
    except Exception as e:
      logging.error(str(e))
      self.ui.vulkanVersionInformation.text = "unknown"

  def onApplicationRestart(self):
    slicer.util.restart()

class PyTorchUtilsLogic(ScriptedLoadableModuleLogic):
  # PyTorch version required by the most recent torchvulkan release, queried from PyPI at most once per session
  _torchvulkanTorchRequirementCache = {}
  # Set to True when the user was already asked whether PyTorch should be upgraded for torchvulkan in this session
  _torchUpgradeOffered = False

  def __init__(self):
    self._torch = None
    self._vulkanDevice = None

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
  def vulkanVersionInformation(self):
    """Get Vulkan driver and torchvulkan information as a string that can be displayed to the user.

    This never installs anything and never accesses the network.
    """
    if not PyTorchUtilsLogic.vulkanDriverAvailable():
      return "not found"
    if PyTorchUtilsLogic.torchvulkanInstalled():
      try:
        import torchvulkan
        if torchvulkan.is_available():
          return f"driver found, torchvulkan {torchvulkan.__version__}: {torchvulkan.get_device_name()}"
        return f"driver found, torchvulkan {torchvulkan.__version__} installed, no usable device"
      except Exception as e:
        logging.warning(str(e))
        return "driver found, torchvulkan cannot be loaded"
    torchRequirement = PyTorchUtilsLogic._torchvulkanTorchRequirementCache.get('value')
    if torchRequirement and not PyTorchUtilsLogic._installedTorchVersionSatisfies(torchRequirement):
      return f"driver found, torchvulkan requires PyTorch {torchRequirement}"
    return "driver found, torchvulkan not installed"

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
    """Import the ``torch`` Python module, installing it if necessary.

    If CUDA is not available but a Vulkan driver is, then the ``vulkan`` device is also registered
    (installing torchvulkan if necessary).
    """
    if PyTorchUtilsLogic.torchInstalled():
      import torch
    else:
      torch = self.installTorch()
    if torch is None:
      logging.warning('PyTorch was not installed')
    else:
      logging.info(f'PyTorch {torch.__version__} imported successfully')
      logging.info(f'CUDA available: {torch.cuda.is_available()}')
      self._initVulkan(torch)
    return torch

  def _initVulkan(self, torch):
    """Register the ``vulkan`` device of PyTorch, installing torchvulkan if necessary.

    CUDA is always preferred. torchvulkan is experimental, it is constrained to specific PyTorch versions and it is
    available only on certain platforms, therefore it is only attempted when CUDA is not available and a Vulkan
    driver is accessible. Any failure is logged and computation simply falls back to the CPU.
    """
    self._vulkanDevice = None
    if torch.cuda.is_available():
      return
    if not PyTorchUtilsLogic.vulkanDriverAvailable():
      logging.info('Vulkan driver not found, therefore the Vulkan backend of PyTorch is not used')
      return
    try:
      torchRequirement = PyTorchUtilsLogic._getTorchvulkanTorchRequirement()
      if torchRequirement and not PyTorchUtilsLogic._installedTorchVersionSatisfies(torchRequirement):
        if not self._offerTorchUpgradeForVulkan(torchRequirement):
          return
        try:
          torch = self.installTorch(askConfirmation=False, torchVersionRequirement=torchRequirement)
        except Exception as e:
          logging.warning(f'Failed to install PyTorch {torchRequirement}: {e}')
          return
        if torch is None:
          return
        self._torch = torch
        # The previously imported torch module is still loaded in this process, so the Vulkan device can only be used
        # after the application has been restarted.
        logging.info(f'PyTorch {torchRequirement} was installed for torchvulkan, the application must be restarted to use the Vulkan device')
        if slicer.util.mainWindow() is not None:
          slicer.util.messageBox(
            f'PyTorch {torchRequirement} was installed so that the Vulkan device of this computer can be used for'
            ' computation. The application must be restarted before PyTorch can use the Vulkan device.')
        return
      if not PyTorchUtilsLogic.installTorchvulkan():
        return
      import torchvulkan
      self._vulkanDevice = torch.device('vulkan')
      logging.info(f'Vulkan device available: {torchvulkan.get_device_name()}')
    except Exception as e:
      logging.warning(f'Failed to initialize the Vulkan backend of PyTorch: {e}')

  def _offerTorchUpgradeForVulkan(self, torchVersionRequirement):
    """Offer to install a PyTorch version that torchvulkan supports, so that the Vulkan device can be used.

    The user is asked only once per session, and only if a user interface is available.

    :param torchVersionRequirement: PyTorch version requirement of torchvulkan (for example ">=2.15.0,<2.16")
    :return: True if the user requested the upgrade
    """
    if PyTorchUtilsLogic._torchUpgradeOffered:
      return False
    PyTorchUtilsLogic._torchUpgradeOffered = True
    if slicer.app.commandOptions().testingEnabled or slicer.util.mainWindow() is None:
      logging.info(
        f'torchvulkan requires PyTorch {torchVersionRequirement}, but there is no user interface to offer installing it')
      return False
    import importlib.metadata
    try:
      installedVersion = importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
      return False
    return slicer.util.confirmOkCancelDisplay(
      f'The Vulkan device of this computer cannot be used for computation, because torchvulkan requires PyTorch'
      f' {torchVersionRequirement} but PyTorch {installedVersion} is installed.'
      f' PyTorch {torchVersionRequirement} will be downloaded and installed if you continue, which enables GPU computation.'
      ' The process might take some minutes, and the application will have to be restarted afterwards.')

  def installTorch(self, askConfirmation=False, forceComputationBackend=None, torchVersionRequirement=None, torchvisionVersionRequirement=None):
    """Install PyTorch and return the ``torch`` Python module.

    :param forceComputationBackend: optional parameter to set computation backend (cpu, cu116, cu117, ...)
    :param torchVersionRequirement: optional version requirement for torch (e.g., ">=1.12")
    :param torchvisionVersionRequirement: optional version requirement for torchvision (e.g., ">=0.8")

    If computation backend is not specified then the ``light-the-torch`` Python package is used to get the most recent version of
    PyTorch compatible with the installed NVIDIA drivers. If CUDA-compatible device is not found, a version compiled for CPU will be installed.
    If no CUDA-compatible device is found but a Vulkan driver is available, then the most recent PyTorch version that
    torchvulkan supports is installed, so that the GPU can be used through Vulkan.
    """

    if torchVersionRequirement is None:
      torchVersionRequirement = PyTorchUtilsLogic._getPreferredTorchVersionRequirement(forceComputationBackend)

    args = PyTorchUtilsLogic._getPipInstallArguments(forceComputationBackend, torchVersionRequirement, torchvisionVersionRequirement)

    if askConfirmation and not slicer.app.commandOptions().testingEnabled:
      install = slicer.util.confirmOkCancelDisplay(
        f'PyTorch will be downloaded and installed using light-the-torch (ltt {" ".join(args)}).'
        ' The process might take some minutes.'
        ' If no CUDA-compatible device is found but a Vulkan driver is available, then the Vulkan backend of PyTorch'
        ' (torchvulkan) will be installed as well.'
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
    PyTorchUtilsLogic.uninstallTorchvulkan()
    slicer.util.pip_uninstall('torch torchvision')
    logging.info(f'PyTorch uninstalled successfully.')

  @staticmethod
  def torchvulkanInstalled():
    """Return True if the ``torchvulkan`` package is installed. The package is not imported."""
    import importlib.metadata
    try:
      importlib.metadata.distribution('torchvulkan')
    except importlib.metadata.PackageNotFoundError:
      return False
    return True

  @staticmethod
  def vulkanDriverAvailable():
    """Return True if a Vulkan driver (Vulkan loader library) is accessible on this system.

    This only checks that the Vulkan loader can be loaded. Whether any GPU can actually be used is verified by
    importing torchvulkan and calling its ``is_available()`` method.
    """
    import sys
    import platform
    if sys.platform == 'darwin':
      # The macOS wheels of torchvulkan bundle MoltenVK, so no system-wide Vulkan driver is needed,
      # but these wheels are built only for Apple Silicon
      return platform.machine() == 'arm64'
    if sys.platform == 'win32':
      import os
      systemRoot = os.environ.get('SystemRoot', r'C:\Windows')
      return os.path.exists(os.path.join(systemRoot, 'System32', 'vulkan-1.dll'))
    # On Linux and other Unix-like systems the Vulkan loader is provided by the GPU driver
    import ctypes
    import ctypes.util
    libraryPath = ctypes.util.find_library('vulkan')
    if not libraryPath:
      return False
    try:
      ctypes.CDLL(libraryPath)
    except OSError:
      return False
    return True

  @staticmethod
  def installTorchvulkan():
    """Make the Vulkan device of PyTorch available and return True if it is usable.

    torchvulkan is installed with a pip constraints file that pins the already installed PyTorch packages, therefore
    it is installed only if it is compatible with the installed PyTorch version and it can never change that version.
    If the installed PyTorch version is incompatible (torchvulkan is released for specific PyTorch versions) or
    torchvulkan is not available for the current platform or Python version, then it is not installed and False is
    returned.
    """
    if not PyTorchUtilsLogic.torchvulkanInstalled():
      torchRequirement = PyTorchUtilsLogic._getTorchvulkanTorchRequirement()
      if torchRequirement and not PyTorchUtilsLogic._installedTorchVersionSatisfies(torchRequirement):
        logging.warning(
          f'torchvulkan requires PyTorch {torchRequirement} but a different PyTorch version is installed, '
          'therefore the Vulkan device cannot be used')
        return False
      logging.info('Installing torchvulkan (Vulkan backend for PyTorch)...')
      constraintFile = PyTorchUtilsLogic._createPipConstraintFile()
      try:
        # --only-binary: torchvulkan is a compiled extension that can be built from source only with a Vulkan SDK
        # and a matching toolchain, so building it locally is never attempted
        slicer.util.pip_install(['torchvulkan', '--only-binary=torchvulkan', '--constraint', constraintFile])
      except Exception as e:
        logging.warning(f'Failed to install torchvulkan with pip: {e}')
      finally:
        import os
        try:
          os.unlink(constraintFile)
        except OSError:
          pass
      if not PyTorchUtilsLogic.torchvulkanInstalled():
        PyTorchUtilsLogic._installTorchvulkanWheel()
      if not PyTorchUtilsLogic.torchvulkanInstalled():
        logging.warning(
          'torchvulkan was not installed: it is available only for certain PyTorch versions, platforms, and Python versions'
        )
        return False
    if PyTorchUtilsLogic._torchvulkanUsable():
      return True
    # Unusable torchvulkan is removed so that it does not prevent installing PyTorch later
    try:
      PyTorchUtilsLogic.uninstallTorchvulkan()
    except Exception as e:
      logging.warning(f'Failed to uninstall torchvulkan: {e}')
    return False

  @staticmethod
  def _getPreferredTorchVersionRequirement(forceComputationBackend=None):
    """Get the PyTorch version requirement to use when PyTorch is installed, or None if there is no preference.

    If a CUDA-compatible device is available then CUDA is preferred and the newest PyTorch version is used. If there is
    no CUDA-compatible device but a Vulkan driver is available, then the newest PyTorch version supported by
    torchvulkan is requested, so that the GPU can be used through Vulkan.

    :param forceComputationBackend: optional parameter to set computation backend (cpu, cu116, cu117, ...)
    """
    if not PyTorchUtilsLogic.vulkanDriverAvailable():
      return None
    try:
      backends = PyTorchUtilsLogic.getCompatibleComputationBackends(forceComputationBackend=forceComputationBackend)
    except Exception as e:
      logging.warning(f'Failed to determine the available computation backends: {e}')
      return None
    if any(backend != 'cpu' for backend in backends):
      # A CUDA-compatible device is available, which is preferred over Vulkan
      return None
    requirement = PyTorchUtilsLogic._getTorchvulkanTorchRequirement()
    if requirement:
      logging.info(
        f'No CUDA-compatible device was found but a Vulkan driver is available, therefore PyTorch {requirement}'
        ' (the version required by torchvulkan) is installed')
    return requirement

  @staticmethod
  def _getTorchvulkanTorchRequirement():
    """Get the PyTorch version requirement of the most recent torchvulkan release.

    :return: PyTorch version requirement (for example ">=2.15.0,<2.16"), or None if torchvulkan does not restrict it
      or it cannot be determined (for example, there is no internet connection)
    """
    cache = PyTorchUtilsLogic._torchvulkanTorchRequirementCache
    if 'value' in cache:
      return cache['value']
    requirement = None
    try:
      import json
      import urllib.request
      from packaging.requirements import Requirement
      with urllib.request.urlopen('https://pypi.org/pypi/torchvulkan/json', timeout=15) as response:
        requiresDist = json.load(response)['info'].get('requires_dist') or []
      for entry in requiresDist:
        dependency = Requirement(entry)
        if dependency.name.lower() == 'torch' and not dependency.marker:
          requirement = str(dependency.specifier) or None
          break
    except Exception as e:
      logging.warning(f'Failed to determine the PyTorch version required by torchvulkan: {e}')
    cache['value'] = requirement
    return requirement

  @staticmethod
  def _installedTorchVersionSatisfies(versionRequirement):
    """Return True if the installed PyTorch version satisfies the given version requirement."""
    import importlib.metadata
    from packaging.specifiers import SpecifierSet
    try:
      installedVersion = importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
      return False
    return SpecifierSet(versionRequirement).contains(installedVersion, prereleases=True)

  @staticmethod
  def _getSystemGlibcVersion():
    """Get the version of the system glibc as a (major, minor) tuple, or None if it cannot be determined."""
    import os
    try:
      version = os.confstr('CS_GNU_LIBC_VERSION')  # for example "glibc 2.28"
    except (ValueError, OSError, AttributeError):
      return None
    parts = version.split()
    if len(parts) != 2 or not parts[0].startswith('glibc'):
      return None
    try:
      major, minor = parts[1].split('.')[:2]
      return (int(major), int(minor))
    except ValueError:
      return None

  @staticmethod
  def _installTorchvulkanWheel():
    """Install torchvulkan by unpacking its wheel, without pip's platform tag check.

    Slicer's Python rejects manylinux wheels that are built for a newer glibc than Slicer's own binaries are built
    for (manylinux_2_28), while torchvulkan publishes manylinux_2_34 wheels for Linux. Such a wheel only requires
    glibc 2.34 on the machine that runs it, so if the system has a recent enough glibc then the matching wheel is
    unpacked into the site-packages directory. If the system glibc is too old then the wheel could not be loaded, so
    nothing is installed.

    :return: True if the torchvulkan wheel was unpacked
    """
    import os
    import platform
    import subprocess
    import sys
    import sysconfig
    import tempfile
    import zipfile
    if not sys.platform.startswith('linux'):
      return False
    architecture = {'x86_64': 'x86_64', 'amd64': 'x86_64', 'aarch64': 'aarch64', 'arm64': 'aarch64'}.get(platform.machine())
    if architecture is None:
      return False
    glibcVersion = PyTorchUtilsLogic._getSystemGlibcVersion()
    if glibcVersion is None or glibcVersion < (2, 34):
      logging.info('The glibc of this system is older than 2.34, therefore the torchvulkan wheel cannot be loaded')
      return False
    pythonVersion = f'{sys.version_info[0]}.{sys.version_info[1]}'
    downloadDirectory = tempfile.mkdtemp(prefix='torchvulkan-wheel-')
    command = [
      sys.executable, '-m', 'pip', 'download', 'torchvulkan', '--no-deps', '--only-binary=:all:',
      '--implementation', 'cp', '--python-version', pythonVersion, '--abi', f'cp{pythonVersion.replace(".", "")}',
      '--platform', f'manylinux_2_34_{architecture}', '--dest', downloadDirectory,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wheelFiles = sorted(f for f in os.listdir(downloadDirectory) if f.endswith('.whl'))
    if result.returncode != 0 or not wheelFiles:
      logging.warning(f'Failed to download the torchvulkan wheel: {result.stdout.strip()}')
      return False
    with zipfile.ZipFile(os.path.join(downloadDirectory, wheelFiles[0])) as wheelFile:
      wheelFile.extractall(sysconfig.get_paths()['purelib'])
    logging.info(f'torchvulkan wheel {wheelFiles[0]} unpacked into {sysconfig.get_paths()["purelib"]}')
    return True

  @staticmethod
  def _createPipConstraintFile():
    """Create a pip constraints file that pins the already installed torch and torchvision packages.

    Passing this file to pip makes it install torchvulkan only if torchvulkan is compatible with the installed
    PyTorch version, and makes it impossible for pip to change the installed PyTorch version.
    """
    import os
    import tempfile
    import importlib.metadata
    constraints = []
    for packageName in ('torch', 'torchvision'):
      try:
        constraints.append(f'{packageName}=={importlib.metadata.version(packageName)}')
      except importlib.metadata.PackageNotFoundError:
        pass
    constraintFile = os.path.join(tempfile.mkdtemp(prefix='torchvulkan-constraints-'), 'constraints.txt')
    with open(constraintFile, 'w') as f:
      f.write('\n'.join(constraints))
    return constraintFile

  @staticmethod
  def uninstallTorchvulkan():
    """Uninstall the ``torchvulkan`` package, if it is installed."""
    if not PyTorchUtilsLogic.torchvulkanInstalled():
      return
    slicer.util.pip_uninstall('torchvulkan')
    logging.info('torchvulkan uninstalled successfully.')

  @staticmethod
  def _torchvulkanUsable():
    """Return True if torchvulkan can be imported and it provides a usable Vulkan device."""
    try:
      import torchvulkan
      if torchvulkan.is_available():
        logging.info(f'torchvulkan {torchvulkan.__version__} is usable, {torchvulkan.device_count()} Vulkan device(s) found')
        return True
      logging.warning('torchvulkan is installed but no usable Vulkan device was found')
    except Exception as e:
      import torch
      logging.warning(f'torchvulkan is not compatible with the installed PyTorch {torch.__version__}: {e}')
    return False

  @staticmethod
  def _getPipInstallArguments(forceComputationBackend=None, torchVersionRequirement=None, torchvisionVersionRequirement=None):
    import sys
    import platform
    macOSIntel = sys.platform == "darwin" and platform.machine() == "x86_64"
    if macOSIntel:
      # macOS with Rosetta2 requires torch >= 2.1.2; torchvision 0.16.2 corresponds to that torch version
      if not torchVersionRequirement:
        torchVersionRequirement = ">=2.1.2"
      if not torchvisionVersionRequirement:
        torchvisionVersionRequirement = ">=0.16.2"
    if sys.platform == "win32":
      # Temporary workaround: torch 2.13 cannot be installed on Windows into a default Slicer installation.
      # This version ships PEP 639 license files that reproduce PyTorch's vendored source tree inside
      # torch-<version>.dist-info/licenses. The deepest collected path is 162 characters relative to
      # site-packages, while Windows limits directory creation to MAX_PATH-12 (248 characters), therefore
      # the installation fails with "[WinError 206] The filename or extension is too long" if the path of
      # site-packages is longer than 86 characters (a default Slicer installation is exactly at this limit).
      # The failed installation leaves a partially extracted torch package behind, which is hard to remove,
      # because deleting those files hits the same path length limit.
      # This restriction can be removed when PyTorch no longer collects the deeply nested license files
      # (see https://github.com/pytorch/pytorch/pull/185813).
      torchVersionRequirement = PyTorchUtilsLogic._addVersionRequirement(torchVersionRequirement, "!=2.13.*")
      # torchvision 0.28 requires torch 2.13, therefore it has to be excluded as well
      torchvisionVersionRequirement = PyTorchUtilsLogic._addVersionRequirement(torchvisionVersionRequirement, "!=0.28.*")
    if torchVersionRequirement is None:
      torchVersionRequirement = ""
    if torchvisionVersionRequirement is None:
      torchvisionVersionRequirement = ""
    args = ["install", "torch"+torchVersionRequirement, "torchvision"+torchvisionVersionRequirement]
    if macOSIntel:
      # On Intel macOS (and Rosetta2), the PyTorch version available on PyPI requires numpy<2, and installing it with pip will fail if numpy 2 is already installed. This is not an issue on other platforms where newer PyTorch versions are available.
      args.append("numpy<2")
    if forceComputationBackend is not None:
      args.append(f"--pytorch-computation-backend={forceComputationBackend}")
    return args

  @staticmethod
  def _addVersionRequirement(versionRequirement, additionalRequirement):
    """Combine a version requirement (such as ">=2.1.2") with an additional requirement (such as "!=2.13.*").

    :param versionRequirement: existing version requirement, may be None or empty
    :param additionalRequirement: version requirement to add
    """
    if not versionRequirement:
      return additionalRequirement
    return versionRequirement + "," + additionalRequirement

  @staticmethod
  def _installLightTheTorch():
    # Install from the Slicer fork which is maintained for newer CUDA versions
    # see: https://github.com/fepegar/SlicerPyTorch/pull/20
    import os
    os.environ["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LIGHT_THE_TORCH"] = "0.8.0"
    slicer.util.pip_install(
      'https://github.com/Slicer/light-the-torch/archive/main.zip')

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
    """Get CUDA device if available, Vulkan device if available, and CPU otherwise."""
    torch = self.torch  # importing torch if necessary, which also registers the Vulkan device if applicable
    if torch.cuda.is_available():
      return torch.device('cuda')
    return self._vulkanDevice if self._vulkanDevice is not None else 'cpu'

  @property
  def cuda(self):
    """Return True if a CUDA-compatible device is available."""
    return self.torch.cuda.is_available()

  @property
  def vulkan(self):
    """Return True if a Vulkan-compatible device is available."""
    return self.getDevice() != 'cpu' and not self.torch.cuda.is_available()


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
