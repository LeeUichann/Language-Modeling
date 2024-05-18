import torch
from torch.utils.data import Dataset
import numpy as np

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file, seq_length=30):
        
        
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()
        
        self.char_to_index = {ch: i for i, ch in enumerate(sorted(set(text)))}
        self.index_to_char = {i: ch for ch, i in self.char_to_index.items()}
        
        self.data = [self.char_to_index[ch] for ch in text]
        
        self.seq_length = seq_length
        self.num_samples = len(self.data) // self.seq_length
        self.input_size = len(self.char_to_index)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length + 1  
        sequence = self.data[start:end]
        
        input_indices = sequence[:-1]
        target_indices = sequence[1:]
        
        input_tensor = torch.tensor(input_indices, dtype=torch.long)
        target_tensor = torch.tensor(target_indices, dtype=torch.long)
        
        return input_tensor, target_tensor

if __name__ == '__main__':
    dataset = Shakespeare(input_file='/data/shakespeare_train.txt')
    print(f"Total sequences: {len(dataset)}")
    print(dataset.input_size)
    for i in range(10):
        input_sample, target_sample = dataset[i]
        print("Input: ", ''.join([dataset.index_to_char[idx] for idx in input_sample.numpy()]))
        print("Target: ", ''.join([dataset.index_to_char[idx] for idx in target_sample.numpy()]))
    
