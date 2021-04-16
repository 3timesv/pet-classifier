import cv2
import utils
from torch.utils.data import DataLoader, Dataset


class PetDataset(Dataset):
    def __init__(self, path_list, label_func, transform=None):

        self.path_list = utils.filter_pathlist(path_list)
        self.label_func = label_func
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):

        image_path = self.path_list[idx]

        image = cv2.imread(str(image_path))

        if image is None:
            return None

        label = self.label_func(image_path.name)

        sample = (image, label)

        if self.transform:
            sample = self.transform(sample)

        return sample[0].float(), sample[1]



class Loader(DataLoader):
    def __init__(self, path_list, label_func, batch_size, shuffle, transform):
        self.ds = PetDataset(path_list, label_func, transform)
        super().__init__(self.ds, batch_size=batch_size, shuffle=shuffle)

    def get_dataset(self):
        return self.ds
