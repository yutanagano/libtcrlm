# libtcrlm

> [!NOTE]
> **To new collaborators that have been added to this repository:**
> You have been added to this repository because you have access to the `sceptr` deployment repo.
> `libtcrlm` (this package) is now a dependency of `sceptr`, and so to install the newest versions of `sceptr` you also need access here.
> **You do not need to manually install this repository**- it will automatically be installed in the background if you install the newest releases of `sceptr`.

This is the TCR language modelling library that powers SCEPTR.
It is a thin layer around PyTorch with some extra infrastructure.

The library is designed only to contain the code that is necessary when deploying our trained models.
By having this code as its own lightweight library, we **remove code duplication between the training/development codebase as well as the deployment codebase**.

This should stay closed-source until SCEPTR is published.