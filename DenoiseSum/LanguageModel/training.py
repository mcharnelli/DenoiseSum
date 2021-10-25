import os

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from DenoiseSum.LanguageModel.data import LanguageModelData

from DenoiseSum.LanguageModel.lm import LM
from DenoiseSum.utils import get_lm_dict, pad, Batcher


def train_language_model(args):
    word_dict = get_lm_dict(args.train_file)
    word_size = len(word_dict)

    x_train = LanguageModelData(args.train_file, word_dict, "text")
    x_dev = LanguageModelData(args.dev_file, word_dict, "text")

    x_train_batcher = Batcher(x_train, args.batch_size)
    x_dev_batcher =  Batcher(x_dev, args.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device('cpu')
    model = LM(word_size, args.word_dim, args.hidden_dim)
    model.to(device)

    optimizer = torch.optim.Adagrad(
        model.parameters(), lr=0.1, initial_accumulator_value=0.1
    )
    best_loss = 1000

    if os.path.exists(args.model_file):
        best_point = torch.load(args.model_file)
        model.load_state_dict(best_point["state_dict"])
        optimizer.load_state_dict(best_point["optimizer"])

    eval_count = args.eval_every
    stop_count = args.stop_after
    losses = []
    for epoch in range(args.num_epoch):
        if stop_count <= 0:
            break

        for batch in tqdm(x_train_batcher):
            if stop_count <= 0:
                x_train_batcher.close()
                break

            model.train()

            x_batch, x_mask = pad(batch)
            x_batch = torch.tensor(x_batch).to(device)
            x_mask = torch.tensor(x_mask).float().to(device)

            batch_loss = model(x_batch, x_mask)
            losses.append(batch_loss.item())
            print(losses)

            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            eval_count -= len(x_batch)
            if eval_count <= 0:
                with torch.no_grad():

                    train_loss = np.mean(losses)
                    dev_loss = []

                    for x_dev_batch in tqdm(x_dev_batcher):
                        model.eval()

                        x_batch, x_mask = pad(x_dev_batch)
                        x_batch = torch.tensor(x_batch).cuda()
                        x_mask = torch.tensor(x_mask).float().cuda()

                        batch_loss = model(x_batch, x_mask)
                        dev_loss.append(batch_loss.item())

                    dev_loss = np.mean(dev_loss)
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

                    tqdm.write(
                        "Epoch: %d, Batch: %d, Train Loss: %.4f, Dev Loss: %.4f"
                        % (epoch, 0, train_loss, dev_loss)
                    )
                    losses = []
                    eval_count = args.eval_every
