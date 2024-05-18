import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers = 1):
        
        # write your codes here
        super(CharRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, 62)
        self.rnn = nn.RNN(62, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.char_to_index = {}
        self.index_to_char = {}

    def forward(self, input, hidden):

        # write your codes here
        embedded = self.embedding(input)
        out, hidden = self.rnn(embedded, hidden)
        batch_size, seq_len, _ = out.size()
        out = out.contiguous().view(batch_size * seq_len, self.hidden_size)
        output = self.fc(out)
        return output, hidden



    def init_hidden(self, batch_size,device):

		# write your codes here
        

        return torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers = 1):

        # write your codes here
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, 62)
        self.lstm = nn.LSTM(62, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):

        # write your codes here
        embedded = self.embedding(input)
        out, hidden = self.lstm(embedded, hidden)
        batch_size, seq_len, _ = out.size()
        out = out.contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.fc(out.contiguous().view(-1, self.hidden_size))
        return output, hidden

    def init_hidden(self, batch_size,device):

		# write your codes here
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        return (h_0, c_0)