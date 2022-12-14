# aiPixels - Stable Diffusion Desktop



Colab API Server added with limited functionality

Please join our Discord for further information: https://discord.gg/FxKSrdrPeW


Welcome to aiPixels, a desktop GUI with Deforum Art, Outpaint, Upscalers, and many more.

A more simple install method is WIP, until then, please make sure that you have a working Anaconda, or Miniconda installation, then run:


```\n
conda env create -n ai-pixel -f environment-installer.yaml
conda activate ai-pixel
setup.bat
```

Once it is finished, the GUI will show up, but before you continue, please make sure that you have your models, model configs and vae files in the models folder named the same. e.g.:
v1-5-pruned-emaonly.ckpt (*model)
v1-5-pruned-emaonly.pt (VAE, optional)
v1-5-pruned-emaonly.yaml (*config)

With RTX3XXX, xFormers is automatically installed, for the rest, if you'd like acceleration, do the following steps:

install MS Visual Studio 2019 with Windows 10 SDK, Cuda 11.6
https://developer.nvidia.com/cuda-11-6-0-download-archive
https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=community&rel=16&utm_medium=microsoft&utm_campaign=download+from+relnotes&utm_content=vs2019ga+button
https://www.anaconda.com/products/distribution

Then:
```\n
conda env create -n aiPixels -f environment_310.yaml
conda activate aiPixels
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
git clone https://github.com/facebookresearch/xformers
cd xformers
(optionally) pip install ninja
git submodule update --init --recursive
pip install -r requirements-test.txt
pip install -e .```

Linux, macOS installers coming up.
