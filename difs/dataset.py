from torch import Tensor
from torch.utils.data import Dataset


class DatasetConditional(Dataset):
    def __init__(self, tensor: Tensor, condition_tensor: Tensor):
        """
        Initializes a dataset with conditions.

        Args:
            tensor (Tensor): The main data tensor.
                The tensor containing the main data.
            condition_tensor (Tensor): The condition tensor.
                The tensor containing the conditional values associated with the main data.

        """
        super().__init__()
        self.tensor = tensor.clone()
        self.condition_tensor = condition_tensor.clone()

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.tensor)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the main data tensor and the associated condition tensor.

        """
        return self.tensor[idx], self.condition_tensor[idx]