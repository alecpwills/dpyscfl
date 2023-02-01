# dpyscfl

## Dependencies

Libnxc depends on PyTorch. The C++/Fortran implementation requires that libtorch is made available at compile time, pylibnxc depends on pytorch.
Both libraries are available for download on the [pytorch website](https://pytorch.org/get-started/locally/).

pylibnxc further depends on numpy, which is a requirement for pytorch and should therefore be installed automatically.

**Optional dependencies:**

For unit testing, pylibnxc currently requires [pyscf<=2.0](https://sunqm.github.io/pyscf/install.html) which can be obtained through

`pip install "pyscf<=2.0"`

Additionally, as of writing, pyscf has not implemented Python 3.10+ pip build wheels for versions of pyscf <=2.1, so **the Python version should be 3.9 or earlier to install a working version of pyscf.**

To unit test the C++/Fortran implementation [GoogleTest](https://github.com/google/googletest) is required.