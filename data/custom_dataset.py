import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, transformations=None):
        self.transformations = transformations
        print(root)
        self.imgs = sorted(glob(f"{root}/*/*.[pj][pn]g"))  # Adjust pattern for all image types

        self.cls_names, self.cls_counts = {}, {}
        count = 0
        for idx, im_path in enumerate(self.imgs):
            class_name = self.get_class(im_path)
            if class_name not in self.cls_names:
                self.cls_names[class_name] = count
                self.cls_counts[class_name] = 1
                count += 1
            else:
                self.cls_counts[class_name] += 1

        print(f"Classes found: {self.cls_names}")
        print(f"Class counts: {self.cls_counts}")

    def get_class(self, path):
        """Extract the class label from the image path."""
        return os.path.dirname(path).split("/")[-1]

    def __len__(self):
        """Return the total number of images."""
        return len(self.imgs)

    def __getitem__(self, idx):
        """Return an image and its corresponding label."""
        im_path = self.imgs[idx]
        im = Image.open(im_path).convert("RGB")
        gt = self.cls_names[self.get_class(im_path)]

        if self.transformations is not None:
            im = self.transformations(im)

        return im, gt

