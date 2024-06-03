# IN THE PARENT DIRECTORY OF THIS GIT REPO:
mkdir qtta_venv
python3 -m venv qtta_venv
pip3 install -r qtta/requirements.txt
git clone https://github.com/huyvnphan/PyTorch_CIFAR10.git
git clone https://github.com/DequanWang/tent.git
git clone git@github.com:mit-han-lab/tinyml.git
git clone git@github.com:mit-han-lab/tiny-training.git 

# curl down the model weights (will have to get my oauth token)
curl -X POST https://content.dropboxapi.com/2/files/download \
    --header "Authorization: Bearer <token here>" \
    --header "Dropbox-API-Arg: {\"path\":\"/resnet18_quant_no_fuse.pth\"}" # or mbn_quant_no_fuse.pth

# download model weights
cd PyTorch_CIFAR10
python3 train.py --download_weights 1

# download dataset
wget https://zenodo.org/records/3555552/files/CIFAR-100-C.tar
