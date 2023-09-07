
import random
import pickle
import torch
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from tqdm import tqdm
from model import _initialize_weights, T5custom
from poly_lang import prepareData
from utils import get_dataloader

MAX_LENGTH = 32 # Maximum sentence length to consider, can be 29, but kept it 32 for <SOS>, <EOS>.
train_file_path  = "train.txt"

if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps") # avoid mps as Pytorch doesnt support all functionalities of my code yet. (torch.roll)
else:
    device = torch.device("cpu") 
print("Device being used:", device)

# Training code for one epoch.
def train_epoch(dataloader, optimizer, criterion):

    total_loss = 0
    input_tensor, target_tensor, pred = None, None, None
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data

        optimizer.zero_grad()

        pred = model(input_tensor, target_tensor)

        B, T, C = pred.shape 
        pred_reshaped = pred.view(B*T, C)
        
        # left shift the target tensor because it basically needs to act as future to the decoder.(Only loss of <SOS> tokens so it is fine.)
        target_tensor = torch.roll(target_tensor, -1)
        target_tensor[:, -1] = 0

        target_tensor_reshaped = target_tensor.view(B*T)
        
        loss = criterion(pred_reshaped, target_tensor_reshaped)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Validation code for one epoch.
@torch.no_grad()
def validate_epoch(dataloader, criterion):
    model.eval()
    total_loss = 0
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data
        pred = model(input_tensor, target_tensor)

        B, T, C = pred.shape 
        pred_reshaped = pred.view(B*T, C)
        
        # left shift the target tensor because it basically needs to act as future to the decoder.(Only loss of <SOS> tokens so it is fine.)
        target_tensor = torch.roll(target_tensor, -1)
        target_tensor[:, -1] = 0
        
        target_tensor_reshaped = target_tensor.view(B*T)
        
        loss = criterion(pred_reshaped, target_tensor_reshaped)

        total_loss += loss.item()
    model.train()
    return total_loss / len(dataloader)

# Main training loop.
def train(train_dataloader, validation_dataloader, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    model.train()
    plot_losses = []
    plot_val_loss = []
    print_loss_total = 0  # Reset every print_every
    print_validation_loss = 0
    plot_loss_total = 0  # Reset every plot_every
    plot_validation_loss_total = 0
    min_validation_loss = 1e5

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss() # ignore pad index, can include argument ignore_index=0

    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.03, total_iters=10) # scheduler drops LR by 0.03 in the first 10 eps.

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(train_dataloader, optimizer, criterion)
        valildation_loss = validate_epoch(validation_dataloader, criterion)

        print_loss_total += train_loss
        plot_loss_total += train_loss
        print_validation_loss += valildation_loss
        plot_validation_loss_total += valildation_loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_validation_loss_avg = print_validation_loss / print_every
            print_loss_total = 0
            print_validation_loss = 0
            print(f"EPOCH = {epoch}, TRAIN LOSS = {print_loss_avg}, VALIDATION LOSS = {print_validation_loss_avg} ")

        if epoch % plot_every == 0: # Maintain variable for plotting losses later.
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_val_loss.append(plot_validation_loss_total / plot_every)
            plot_validation_loss_total = 0
            plot_loss_total = 0
        
        if min_validation_loss >= valildation_loss:
            min_validation_loss = valildation_loss
            torch.save(model.state_dict(), "weights/newrun_best_model_state_dict.pt")
            
        scheduler.step()
    
    return plot_losses, plot_val_loss


# Language construction and getting the data
input_lang, output_lang, pairs = prepareData(train_file_path)
input_lang = output_lang # We are using the same language for both input and output. (This is not a mistake.)

# Shuffle pairs
random.shuffle(pairs)

# splitting the data into train and test
train_test_split = 0.95
train_set = pairs[:int(len(pairs)*train_test_split)]
validation_set = pairs[int(len(pairs)*train_test_split):]
print("train set : validation set = ", len(train_set), ":", len(validation_set))

# Construct dataloaders
batch_size = 1024
train_dataloader = get_dataloader(batch_size=batch_size, input_lang=input_lang, output_lang=output_lang, pairs=train_set, device=device) # Small subset for testing.
validation_dataloader = get_dataloader(batch_size=batch_size, input_lang=input_lang, output_lang=output_lang, pairs=validation_set, device=device) # Small subset for testing.

# Construct the model
model = T5custom(
        dim = 128, 
        max_seq_len = MAX_LENGTH, # 32 - Max length of the sequence.
        enc_num_tokens = input_lang.n_words, # 36 - count of words in our arbitrary language.
        enc_depth = 3, # count of encoder blocks
        enc_heads = 8, # count of attention heads
        enc_dim_head = 64, # dimension of each attention head
        enc_mlp_mult = 4, # factor to increase the hidden dimension of the MLPs in the encoder
        dec_num_tokens = input_lang.n_words, # 36 - count of words in our arbitrary language.
        dec_depth = 3, # count of decoder blocks
        dec_heads = 8, # count of attention heads
        dec_dim_head = 64, # dimension of each attention head
        dec_mlp_mult = 4, # factor to increase the hidden dimension of the MLPs in the decoder
        dropout = 0.1, # dropout probability
        tie_token_emb = True # whether to tie the weights of the token embeddings.
    )
model.apply(_initialize_weights)
model = model.to(device)

# Train & Validate model
train_losses, val_losses = train(train_dataloader, validation_dataloader, n_epochs=30, learning_rate=0.001, print_every=1, plot_every=1)

# Save model weights
torch.save(model.state_dict(), "weights/newrun_model_state_dict.pt")

# Save language encodings.
with open('weights/newrun_poly_language.pkl', 'wb') as outp:
    pickle.dump(input_lang, outp, pickle.HIGHEST_PROTOCOL)