import os

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from DenoiseSum.LanguageModel.data import LanguageModelData

from DenoiseSum.LanguageModel.lm import LM
from DenoiseSum.utils import get_lm_dict, pad, Batcher

import pickle
import os.path

from torch.utils.data import DataLoader

def eval_model(model, x_dev_batcher, device):
    with torch.no_grad():
        dev_loss = []

        for x_dev_batch in tqdm(x_dev_batcher):
            model.eval()

            x_batch, x_mask = pad(x_dev_batch)
            x_batch = torch.tensor(x_batch).to(device)
            x_mask = torch.tensor(x_mask).float().to(device)

            batch_loss = model(x_batch, x_mask)
            dev_loss.append(batch_loss.item())

        dev_loss = np.mean(dev_loss)
        
    return dev_loss

def train_epoch(model, optimizer, epoch, x_train_batcher, x_dev_batcher, device):
    losses = []
    bar = tqdm(x_train_batcher)
    bar.set_description(f"Epoch {epoch}")
    for i_batch, batch in enumerate(bar):

        batch_loss = train_batch(model, optimizer, batch, device)
        losses.append(batch_loss)
    
    train_loss = np.mean(losses)
    dev_loss = eval_model(model, x_dev_batcher, device)
    print(
        "Epoch: %d, Train Loss: %.4f, Dev Loss: %.4f"
        % (epoch, train_loss, dev_loss)
    )
    
    return train_loss, dev_loss

    

def train_batch(model, optimizer, batch, device):

    model.train()

    x_batch, x_mask = pad(batch)
    x_batch = torch.tensor(x_batch).to(device)
    x_mask = torch.tensor(x_mask).float().to(device)

    batch_loss = model(x_batch, x_mask)

    batch_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 3)
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

def init_word_directory(train_file, replace_word_dict):
    folder = train_file.parent
    file_name = train_file.name
    word_dict_file = file_name.split(".")[0] + "word_dict.pickle"
    word_dict_file = word_dict_file.replace("train", "")
    word_dict_file = folder / word_dict_file
    print(train_file)

    if not replace_word_dict and os.path.exists(word_dict_file):
        with open(word_dict_file, "rb") as handle:
            word_dict = pickle.load(handle)

    else:
        word_dict = get_lm_dict(train_file)
        with open(word_dict_file, "wb") as handle:
            pickle.dump(word_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return word_dict

def train_language_model(args):
    train_losses=[]
    validation_losses=[]
    
    word_dict = init_word_directory(args.train_file, args.replace_word_dict)
    word_size = len(word_dict)

    x_train = LanguageModelData(args.train_file, word_dict, "text")
    x_dev = LanguageModelData(args.dev_file, word_dict, "text")

    x_train_batcher = Batcher(x_train.data1, args.batch_size)
    x_dev_batcher = Batcher(x_dev.data1, args.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LM(word_size, args.word_dim, args.hidden_dim)
    model.to(device)

    optimizer = torch.optim.Adagrad(
        model.parameters(), lr=0.1, initial_accumulator_value=0.1
    )
    best_loss = 1000
    
    if args.retrain_model and os.path.exists(args.model_file):
        best_point = torch.load(args.model_file)
        model.load_state_dict(best_point["state_dict"])
        optimizer.load_state_dict(best_point["optimizer"])

    eval_count = args.eval_every
    stop_count = args.stop_after
    for epoch in range(args.num_epoch):
        if stop_count <= 0:
            break
        train_loss, dev_loss = train_epoch(model, optimizer, epoch, x_train_batcher, x_dev_batcher, device)

        # log losses
        train_losses.append(train_loss)
        validation_losses.append(dev_loss)

        # early stopping
        if best_loss >= dev_loss:
                best_loss = dev_loss
                stop_count = args.stop_after
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    args.model_file,
                )
        else:
            stop_count -= 1

        eval_count = args.eval_every

    # Save the log of the train and validation losses
    folder = args.model_file.parent
    with open(folder / 'train_losses.pickle', "wb") as handle:
        pickle.dump(train_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder / 'validation_losses.pickle', "wb") as handle:
        pickle.dump(validation_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)


