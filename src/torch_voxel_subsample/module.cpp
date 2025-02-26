#include "voxel_subsample.h"
#include "voxel_subsample_deterministic.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxel_subsample", &random_voxel_subsample::voxel_subsample);
    m.def("voxel_subsample_deterministic", &deterministic_voxel_subsample::voxel_subsample);
}
