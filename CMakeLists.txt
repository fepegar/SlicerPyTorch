cmake_minimum_required(VERSION 3.13.4)

project(PyTorch)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/PyTorch")
set(EXTENSION_CATEGORY "")
set(EXTENSION_CONTRIBUTORS "Fernando Perez-Garcia (University College London and King's College London)")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/PyTorch.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/PyTorch/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(PyTorchUtils)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
