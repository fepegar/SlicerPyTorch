
# Torchvulkan (Vulkan) fallback tests. No GPU, Vulkan driver, network access, or installed PyTorch is required,
# because all of them are mocked.
slicer_add_python_unittest(SCRIPT ${MODULE_NAME}VulkanTest.py)
