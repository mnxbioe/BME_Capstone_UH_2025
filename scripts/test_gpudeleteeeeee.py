import torch, platform
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("compiled with cuda:", torch.version.cuda)
print("num gpus:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
print("platform:", platform.platform())
