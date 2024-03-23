# libtcrlm

This is the TCR language modelling library that powers SCEPTR.
It is a thin layer around PyTorch with some extra infrastructure.

The library is designed only to contain the code that is necessary when deploying our trained models.
By having this code as its own lightweight library, we **remove code duplication between the training/development codebase as well as the deployment codebase**.

This should stay closed-source until SCEPTR is published.

## Installation

> [!NOTE]
> This is a library package.
> If you would like to download and use trained SCEPTR models, please refer to the `sceptr` deployment repository.
> The only time it would be useful to download this library is if you would like to use the infrastructure to engineer/train your own TCR language models.

From your python environment, run the following, replacing `<VERSION_TAG>` with the appropriate version specifier.
The latest release tags can be found by checking the 'releases' section on the github repository page.

```bash
pip install git+https://github.com/yutanagano/libtcrlm.git@<VERSION_TAG>
```
