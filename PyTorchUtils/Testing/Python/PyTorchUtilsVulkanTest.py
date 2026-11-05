"""Tests for the Vulkan (torchvulkan) fallback of PyTorchUtils.

No GPU, Vulkan driver, network access, or installed PyTorch is required: hardware, network, and torch/torchvulkan are
mocked. The test is registered as a ctest by ``PyTorchUtils/Testing/Python/CMakeLists.txt`` and runs nightly.
"""

import os
import platform
import shutil
import sys
import tempfile
import types
from unittest import mock

import PyTorchUtils
import slicer

from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest


class _FakeTorch:
  """Minimal stand-in for the ``torch`` module, so that PyTorch does not have to be installed."""

  def __init__(self, cudaAvailable):
    self.cuda = types.SimpleNamespace(is_available=lambda: cudaAvailable)

  def device(self, name):
    return name


class PyTorchUtilsVulkanTest(ScriptedLoadableModuleTest):

  def setUp(self):
    self.Logic = PyTorchUtils.PyTorchUtilsLogic
    # The queried torchvulkan requirement and the "already asked" flag are cached for the whole application session
    self.Logic._torchvulkanTorchRequirementCache.clear()
    self.Logic._torchUpgradeOffered = False

  def test_vulkanDriverDetection(self):
    """A Vulkan driver can be detected on every supported platform without loading any Vulkan library."""
    Logic = self.Logic

    with mock.patch.object(sys, 'platform', 'darwin'), mock.patch.object(platform, 'machine', return_value='arm64'):
      self.assertTrue(Logic.vulkanDriverAvailable(), "torchvulkan bundles MoltenVK on Apple Silicon")

    with mock.patch.object(sys, 'platform', 'darwin'), mock.patch.object(platform, 'machine', return_value='x86_64'):
      self.assertFalse(Logic.vulkanDriverAvailable(), "torchvulkan has no wheel for Intel Macs")

    with tempfile.TemporaryDirectory() as temporaryDirectory:
      os.makedirs(os.path.join(temporaryDirectory, 'System32'))
      with mock.patch.object(sys, 'platform', 'win32'), mock.patch.dict(os.environ, {'SystemRoot': temporaryDirectory}):
        self.assertFalse(Logic.vulkanDriverAvailable(), "no vulkan-1.dll")
      open(os.path.join(temporaryDirectory, 'System32', 'vulkan-1.dll'), 'w').close()
      with mock.patch.object(sys, 'platform', 'win32'), mock.patch.dict(os.environ, {'SystemRoot': temporaryDirectory}):
        self.assertTrue(Logic.vulkanDriverAvailable(), "vulkan-1.dll is present")

  def test_torchVersionSelection(self):
    """The PyTorch version to install is only restricted when Vulkan is the only GPU path."""
    Logic = self.Logic

    with mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: False)):
      self.assertIsNone(Logic._getPreferredTorchVersionRequirement(), "no Vulkan driver, no restriction")

    with mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: True)), \
         mock.patch.object(Logic, 'getCompatibleComputationBackends', staticmethod(lambda **kwargs: ['cpu', 'cu126'])):
      self.assertIsNone(Logic._getPreferredTorchVersionRequirement(), "CUDA is preferred over Vulkan")

    with mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: True)), \
         mock.patch.object(Logic, 'getCompatibleComputationBackends', staticmethod(lambda **kwargs: ['cpu'])), \
         mock.patch.object(Logic, '_getTorchvulkanTorchRequirement', staticmethod(lambda: '>=2.15.0,<2.16')):
      self.assertEqual(Logic._getPreferredTorchVersionRequirement(), '>=2.15.0,<2.16')

    with mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: True)), \
         mock.patch.object(Logic, 'getCompatibleComputationBackends', staticmethod(lambda **kwargs: ['cpu'])), \
         mock.patch.object(Logic, '_getTorchvulkanTorchRequirement', staticmethod(lambda: None)):
      self.assertIsNone(Logic._getPreferredTorchVersionRequirement(), "torchvulkan does not restrict PyTorch")

  def test_deviceSelection(self):
    """getDevice() returns cuda, then vulkan, then cpu."""
    Logic = self.Logic
    logic = Logic()
    logic._torch = _FakeTorch(cudaAvailable=False)

    self.assertEqual(logic.getDevice(), 'cpu')
    self.assertFalse(logic.cuda)
    self.assertFalse(logic.vulkan)

    logic._vulkanDevice = 'vulkan'
    self.assertEqual(logic.getDevice(), 'vulkan')
    self.assertTrue(logic.vulkan)
    self.assertFalse(logic.cuda)

    logic._torch = _FakeTorch(cudaAvailable=True)
    logic._vulkanDevice = 'vulkan'
    self.assertEqual(logic.getDevice(), 'cuda', "CUDA is preferred over Vulkan")
    self.assertTrue(logic.cuda)
    self.assertFalse(logic.vulkan)

  def test_enableVulkan(self):
    """Enabling Vulkan does not install anything when it cannot help, and registers the device when it can."""
    Logic = self.Logic
    logic = Logic()
    logic._torch = _FakeTorch(cudaAvailable=False)

    with mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: False)), \
         mock.patch.object(Logic, 'installTorchvulkan') as installTorchvulkan:
      self.assertFalse(logic.enableVulkan(), "no Vulkan driver")
      self.assertEqual(installTorchvulkan.call_count, 0)
      self.assertIsNone(logic._vulkanDevice)

    with mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: True)), \
         mock.patch.object(Logic, 'installTorchvulkan', staticmethod(lambda: False)):
      self.assertFalse(logic.enableVulkan(), "torchvulkan is unusable")
      self.assertEqual(str(logic.getDevice()), 'cpu')

    with mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: True)), \
         mock.patch.object(Logic, 'installTorchvulkan', staticmethod(lambda: True)):
      self.assertTrue(logic.enableVulkan())
      self.assertEqual(str(logic.getDevice()), 'vulkan')

    cudaLogic = Logic()
    cudaLogic._torch = _FakeTorch(cudaAvailable=True)
    with mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: True)), \
         mock.patch.object(Logic, 'installTorchvulkan') as installTorchvulkan:
      self.assertFalse(cudaLogic.enableVulkan(), "CUDA is available")
      self.assertEqual(installTorchvulkan.call_count, 0)

  def test_getTorchUpgradeRequirementForVulkan(self):
    """An upgrade is only requested when the installed PyTorch cannot work with torchvulkan."""
    Logic = self.Logic
    logic = Logic()

    with mock.patch.object(Logic, 'torchInstalled', staticmethod(lambda: False)), \
         mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: True)):
      self.assertIsNone(logic.getTorchUpgradeRequirementForVulkan(), "PyTorch will be installed, not upgraded")

    with mock.patch.object(Logic, 'torchInstalled', staticmethod(lambda: True)), \
         mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: False)):
      self.assertIsNone(logic.getTorchUpgradeRequirementForVulkan(), "no Vulkan driver")

    with mock.patch.object(Logic, 'torchInstalled', staticmethod(lambda: True)), \
         mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: True)), \
         mock.patch.object(Logic, '_getTorchvulkanTorchRequirement', staticmethod(lambda: '>=2.15.0,<2.16')), \
         mock.patch.object(Logic, '_installedTorchVersionSatisfies', staticmethod(lambda requirement: True)):
      self.assertIsNone(logic.getTorchUpgradeRequirementForVulkan(), "already compatible")

    with mock.patch.object(Logic, 'torchInstalled', staticmethod(lambda: True)), \
         mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: True)), \
         mock.patch.object(Logic, '_getTorchvulkanTorchRequirement', staticmethod(lambda: '>=2.15.0,<2.16')), \
         mock.patch.object(Logic, '_installedTorchVersionSatisfies', staticmethod(lambda requirement: False)):
      self.assertEqual(logic.getTorchUpgradeRequirementForVulkan(), '>=2.15.0,<2.16')

  def test_vulkanVersionInformation(self):
    """The status text shown in the module widget describes the Vulkan state."""
    Logic = self.Logic

    with mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: False)):
      self.assertEqual(Logic().vulkanVersionInformation, 'not found')

    with mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: True)), \
         mock.patch.object(Logic, 'torchvulkanInstalled', staticmethod(lambda: False)):
      self.assertEqual(Logic().vulkanVersionInformation, 'driver found, torchvulkan not installed')

    fakeTorchvulkan = types.ModuleType('torchvulkan')
    fakeTorchvulkan.__version__ = '0.0.0-test'
    fakeTorchvulkan.is_available = lambda: True
    fakeTorchvulkan.get_device_name = lambda: 'Test Vulkan GPU'
    with mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: True)), \
         mock.patch.object(Logic, 'torchvulkanInstalled', staticmethod(lambda: True)), \
         mock.patch.dict(sys.modules, {'torchvulkan': fakeTorchvulkan}):
      self.assertEqual(Logic().vulkanVersionInformation, 'driver found, torchvulkan 0.0.0-test: Test Vulkan GPU')

      fakeTorchvulkan.is_available = lambda: False
      self.assertEqual(Logic().vulkanVersionInformation,
                       'driver found, torchvulkan 0.0.0-test installed, no usable device')

    with mock.patch.object(Logic, 'vulkanDriverAvailable', staticmethod(lambda: True)), \
         mock.patch.object(Logic, 'torchvulkanInstalled', staticmethod(lambda: False)), \
         mock.patch.object(Logic, '_installedTorchVersionSatisfies', staticmethod(lambda requirement: False)):
      Logic._torchvulkanTorchRequirementCache['value'] = '>=2.15.0,<2.16'
      self.assertEqual(Logic().vulkanVersionInformation, 'driver found, torchvulkan requires PyTorch >=2.15.0,<2.16')

  def test_pipConstraintFile(self):
    """The constraints file pins the installed PyTorch packages, so that pip cannot change PyTorch."""
    constraintFile = self.Logic._createPipConstraintFile()
    try:
      self.assertTrue(os.path.isfile(constraintFile))
      with open(constraintFile) as file:
        constraints = file.read().splitlines()
      import importlib.metadata
      for constraint in constraints:
        packageName, version = constraint.split('==')
        self.assertEqual(version, importlib.metadata.version(packageName))
        self.assertIn(packageName, ('torch', 'torchvision'))
    finally:
      shutil.rmtree(os.path.dirname(constraintFile), ignore_errors=True)

  def test_loadedTorchVersionMatchesInstalled(self):
    """An updated PyTorch is only used after a restart, so torchvulkan is not loaded in the same session."""
    Logic = self.Logic
    self.assertTrue(Logic._loadedTorchVersionMatchesInstalled())

    with mock.patch.object(Logic, 'torchvulkanInstalled', staticmethod(lambda: False)), \
         mock.patch.object(Logic, '_getTorchvulkanTorchRequirement', staticmethod(lambda: '>=2.15.0,<2.16')), \
         mock.patch.object(Logic, '_installedTorchVersionSatisfies', staticmethod(lambda requirement: True)), \
         mock.patch.object(Logic, '_loadedTorchVersionMatchesInstalled', staticmethod(lambda: False)), \
         mock.patch.object(PyTorchUtils.slicer.util, 'pip_install') as pip_install:
      self.assertFalse(Logic.installTorchvulkan(), "restart is required after a PyTorch update")
      self.assertEqual(pip_install.call_count, 0, "torchvulkan must not be installed before the restart")

  def test_pythonExecutableForWheelInstall(self):
    """A working interpreter is found even if sys.executable is not a Python interpreter."""
    import shutil
    Logic = self.Logic
    self.assertIsNotNone(Logic._getPythonExecutable())
    if shutil.which('PythonSlicer') is None:
      self.skipTest('PythonSlicer is not in PATH, therefore the fallback cannot be tested')
    with mock.patch.object(sys, 'executable', '/nonexistent/SlicerApp-real'):
      self.assertIsNotNone(Logic._getPythonExecutable(), "PythonSlicer is found through PATH")

  def test_torchvulkanWheelHelpers(self):
    """The unpacked wheel is only used when it matches the installed PyTorch version."""
    import zipfile
    Logic = self.Logic

    platformTags = Logic._getTorchvulkanWheelPlatformTags('x86_64')
    self.assertEqual(platformTags[0], 'manylinux_2_34_x86_64')
    self.assertTrue(all(tag.endswith('_x86_64') for tag in platformTags))

    def writeWheel(directory, metadataText):
      wheelPath = os.path.join(directory, 'torchvulkan-0.0.0-cp312-cp312-manylinux_2_34_x86_64.whl')
      with zipfile.ZipFile(wheelPath, 'w') as wheelFile:
        wheelFile.writestr('torchvulkan-0.0.0.dist-info/METADATA', metadataText)
      return wheelPath

    with tempfile.TemporaryDirectory() as temporaryDirectory:
      wheelPath = writeWheel(temporaryDirectory, 'Metadata-Version: 2.4\nName: torchvulkan\nRequires-Dist: torch<2.16,>=2.15.0\n')
      with mock.patch.object(Logic, '_installedTorchVersionSatisfies', staticmethod(lambda requirement: True)):
        self.assertTrue(Logic._torchvulkanWheelMatchesInstalledTorch(wheelPath))
      with mock.patch.object(Logic, '_installedTorchVersionSatisfies', staticmethod(lambda requirement: False)):
        self.assertFalse(Logic._torchvulkanWheelMatchesInstalledTorch(wheelPath), "wrong PyTorch version")

      unrestrictedWheel = writeWheel(temporaryDirectory, 'Metadata-Version: 2.4\nName: torchvulkan\n')
      self.assertTrue(Logic._torchvulkanWheelMatchesInstalledTorch(unrestrictedWheel),
                      "no PyTorch restriction means compatible")

  def test_torchUpgradeOffer(self):
    """The user is asked at most once per session, and never without a user interface."""
    if not slicer.app.commandOptions().testingEnabled:
      self.skipTest("The test would display a confirmation dialog")
    Logic = self.Logic
    logic = Logic()

    with mock.patch.object(Logic, 'torchInstalled', staticmethod(lambda: True)):
      self.assertFalse(logic._offerTorchUpgradeForVulkan('>=2.15.0,<2.16'), "no dialog in testing mode")
      self.assertTrue(Logic._torchUpgradeOffered)
      self.assertFalse(logic._offerTorchUpgradeForVulkan('>=2.15.0,<2.16'), "the user is asked only once")
