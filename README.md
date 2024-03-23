# libtcrlm

> [!NOTE]
> **To new collaborators that have been added to this repository:**
> You have been added to this repository because you have access to the `sceptr` repository.
> `libtcrlm` (this package) is now a dependency of `sceptr`, and so to install the newest deployments of `sceptr` you need access to this private repository as well.
> In summary, you have been given access so that you can keep using the newest versions of `sceptr`.

This is the TCR language modelling library that powers SCEPTR.
It is a thin layer around PyTorch with some extra infrastructure.

The library is designed only to contain the code that is necessary when deploying our trained models.
By having this code as its own lightweight library, we **remove code duplication between the training/development codebase as well as the deployment codebase**.

This should stay closed-source until SCEPTR is published.

## Installation

From your python environment, run the following, replacing `<VERSION_TAG>` with the appropriate version specifier.
The latest release tags can be found by checking the 'releases' section on the github repository page.

```bash
pip install git+https://github.com/yutanagano/libtcrlm.git@<VERSION_TAG>
```
