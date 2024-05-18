import dataset
from model import CharRNN, CharLSTM

# import some packages you need here
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from dataset import Shakespeare
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import wandb
import json
    




def save_model(model, path):
    torch.save(model.state_dict(), path)

def train(model, train_dl, criterion, optimizer,device):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    # write your codes here
    model.train()
    total_loss = 0

    for inputs, targets in train_dl:
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = model.init_hidden(inputs.size(0),device)

        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
    
    return total_loss / len(train_dl.dataset)


def validate(model, val_dl, criterion, device):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    # write your codes here
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in val_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0),device)
            outputs, hidden = model(inputs, hidden)

            loss = criterion(outputs, targets.view(-1))
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_loss / total_samples




def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    #write your codes here
    
    #wandb.init(project="char_rnn_project", name='rnn_loss')
    dataset = Shakespeare(input_file='/data/shakespeare_train.txt')
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.9 * dataset_size))

    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
   
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_dl = DataLoader(dataset, batch_size=512, sampler = train_sampler)
    val_dl = DataLoader(dataset, batch_size=512, sampler = val_sampler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    

    rnn_model = CharRNN(input_size=dataset.input_size, hidden_size=128, output_size=dataset.input_size, n_layers=1).to(device)
    rnn_model.char_to_index = dataset.char_to_index
    rnn_model.index_to_char = dataset.index_to_char
    lstm_model = CharLSTM(input_size=dataset.input_size, hidden_size=128, output_size=dataset.input_size, n_layers=1).to(device)

    
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    

    criterion = nn.CrossEntropyLoss().to(device)
    
    print('RNN')
    for epoch in range(100):
        train_loss = train(rnn_model, train_dl, criterion, rnn_optimizer, device)
        val_loss = validate(rnn_model, val_dl, criterion, device)
        
        wandb.log({"RNN Train Loss": train_loss, "RNN Val Loss": val_loss})  
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    with open('/nas/home/uichan/hw3/char_to_index.json', 'w') as f:
            json.dump(rnn_model.char_to_index, f)
    with open('/nas/home/uichan/hw3/index_to_char.json', 'w') as f:
             json.dump(rnn_model.index_to_char, f)
    torch.save(model.state_dict(), '/nas/home/uichan/hw3/rnn_model2.pth')
    
    print('LSTM')
    for epoch in range(100):
        train_loss = train(lstm_model, train_dl, criterion, lstm_optimizer, device)
        val_loss = validate(lstm_model, val_dl, criterion, device)
        
        wandb.log({"LSTM Train Loss": train_loss, "LSTM Val Loss": val_loss})  
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    with open('char_to_index.json', 'w') as f:
            json.dump(model.char_to_index, f)
    with open('index_to_char.json', 'w') as f:
             json.dump(model.index_to_char, f)
    torch.save(lstm_model.state_dict(), '/nas/home/uichan/hw3/lstm_model.pth')
    
    wandb.finish()

    

if __name__ == '__main__':
    main()