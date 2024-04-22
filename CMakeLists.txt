cmake_minimum_required(VERSION 3.13.4)

project(PyTorch)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/fepegar/SlicerPyTorch")
set(EXTENSION_CATEGORY "Utilities")
set(EXTENSION_CONTRIBUTORS "Fernando Pérez-García (University College London and King's College London), Andras Lasso (PerkLab Queen's University)")
set(EXTENSION_DESCRIPTION "Utilities to install and use PyTorch within 3D Slicer.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/fepegar/SlicerPyTorch/master/PyTorch.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/fepegar/SlicerPyTorch/master/project_week_diagram.png")
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
