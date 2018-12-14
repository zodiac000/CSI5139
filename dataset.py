import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
from glob import glob

classes = {
        'daisy'       : 0,   
        'dandelion'   : 1,
        'rose'        : 2,
        'sunflower'   : 3,
        'tulip'       : 4,
        }

class Flowers(data.Dataset):
    def __init__(self, is_train, downsample, upsample):
        super().__init__()
        cls = "train" if is_train else "test"
        self.filelist = glob("data/flowers/"+ cls +"/*/*.jpg")
        transforms = []
        if downsample:
            transforms.append(Resize(224//4))
            if upsample:
                transforms.append(Resize(224))
        transforms.append(ToTensor())
        self.transform = Compose(transforms)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img = Image.open(self.filelist[index]).convert("YCbCr").split()[0]
        label = classes[self.filelist[index].split('/')[-2]]
        return self.transform(img), label
