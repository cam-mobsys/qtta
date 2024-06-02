# IN THE PARENT DIRECTORY OF THIS GIT REPO:
mkdir qtta
python3 -m venv qtta
pip3 install -r qtta/requirements.txt
git clone https://github.com/huyvnphan/PyTorch_CIFAR10.git
git clone https://github.com/DequanWang/tent.git
git clone 

# curl down the model weights (will have to get my oauth token)
curl -X POST https://content.dropboxapi.com/2/files/download \
    --header "Authorization: Bearer <token here>" \
    --header "Dropbox-API-Arg: {\"path\":\"/resnet18_quant_no_fuse.pth\"}" # or mbn_quant_no_fuse.pth


