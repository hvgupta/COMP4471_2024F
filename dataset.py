import os
from PIL import Image
from torch.utils.data import Dataset

class SIDD_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.scene_instances = self._load_scene_instances()

    def _load_scene_instances(self):
        scene_instance_file = os.path.join(self.data_dir, 'Scene_Instances.txt')
        with open(scene_instance_file, 'r') as f:
            return f.read().splitlines()

    def __len__(self):
        return len(self.scene_instances)

    def __getitem__(self, idx):
        scene_instance = self.scene_instances[idx]
        folder_path = os.path.join(self.data_dir, "Data", scene_instance)

        noisy_image_path = os.path.join(folder_path, 'NOISY_SRGB_010.png')
        clean_image_path = os.path.join(folder_path, 'GT_SRGB_010.png')

        noisy_image = Image.open(noisy_image_path).convert('RGB')
        clean_image = Image.open(clean_image_path).convert('RGB')

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return clean_image, noisy_image