import torch 

    def get_default_device():
        """Pick GPU if available, else CPU"""
       # cuda0 = torch.device("cuda:0")
       # cuda1 = torch.device("cuda:1")
        cuda = torch.device("cuda")
       # cudas = [cuda0,cuda1]
       # selected_gpu = cudas[random.choice([1,0])]
       # print("selected gpu is ", selected_gpu)
        if torch.cuda.is_available():
            return cuda
        else:
            return torch.device('cpu')



    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)


    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""

        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl:
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)