
# import some packages you need here
from model import CharLSTM
import torch
import json
def generate(model, seed_characters_list, t,length, device):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

    Returns:
        samples: generated characters
    """

    # write your codes here
    model.eval()  
    samples = []

    for seed_characters in seed_characters_list:
        generated = seed_characters

        
        input_indices = [model.char_to_index[char] for char in seed_characters]
        input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
        
        
        hidden = model.init_hidden(1, device)

        
        for char in seed_characters[:-1]:
            _, hidden = model(input_tensor[:, :1], hidden)
            input_tensor = input_tensor[:, 1:]

        input_tensor = input_tensor[:, -1:]

        for _ in range(length):
            output, hidden = model(input_tensor, hidden)
            output = output / t
            output_pro = t_softmax(output, t)
            next_index = torch.multinomial(output_pro, 1).item()
            
            next_char = model.index_to_char[next_index]
       
          
            

            generated += next_char

            input_tensor = torch.tensor([[next_index]], dtype=torch.long).to(device)

        samples.append(generated)
    return samples

def t_softmax(logit, t):

    scaled = logit / t
    output = torch.nn.functional.softmax(scaled, dim=-1)

    return output


if __name__ == '__main__':
 # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CharLSTM(input_size=62, hidden_size=128, output_size=62, n_layers=1)
    model.load_state_dict(torch.load('/nas/home/uichan/hw3/lstm_model.pth'))
    model.to(device)

   
    with open('/nas/home/uichan/hw3/char_to_index.json', 'r') as f:
        model.char_to_index = json.load(f)
    with open('/nas/home/uichan/hw3/index_to_char.json', 'r') as f:
        model.index_to_char = json.load(f)

    model.index_to_char = {int(k): v for k, v in model.index_to_char.items()}
    print(model.char_to_index)
    print(model.index_to_char)
    
    seeds = [
        "In the dead of night",
        "A whisper in the wind",
        "The last light of day",
        "Through the looking glass",
        "Beyond the distant horizon"
    ]


    # Generate samples
    for t in [0.5, 0.8, 1.0, 1.2]:
        print(f'\nTemperature: {t}')
        samples = generate(model, seeds, t=t, length=100, device=device)
        for i, sample in enumerate(samples):
            print(f"\nSample {i+1}:\n{sample}\n{'-'*50}")

