import json
import math
import time

import matplotlib.pyplot as plt
import seaborn
import torch


def load_coco_captions(path: str = "./datasets/coco.pt"):
    """
    Download and load serialized COCO data from coco.pt.

    It contains a dictionary of:
    "train_images" - resized training images (112x112)
    "val_images" - resized validation images (112x112)
    "train_captions" - tokenized and numericalized training captions
    "val_captions" - tokenized and numericalized validation captions
    "vocab" - caption vocabulary, including "idx_to_token" and "token_to_idx"

    Returns: a data dictionary
    """
    data_dict = torch.load(path)

    # Print all keys and values from the data dictionary.
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            print(k, type(v), v.shape, v.dtype)
        else:
            print(k, type(v), v.keys())

    assert data_dict["train_images"].size(0) == data_dict[
        "train_captions"
    ].size(0) and data_dict["val_images"].size(0) == data_dict[
        "val_captions"
    ].size(0), "shapes of data mismatch!"

    print("\nTrain images shape: ", data_dict["train_images"].shape)
    print("Train caption tokens shape: ", data_dict["train_captions"].shape)
    print("Validation images shape: ", data_dict["val_images"].shape)
    print("Validation caption tokens shape: ", data_dict["val_captions"].shape)
    print(
        "total number of caption tokens: ",
        len(data_dict["vocab"]["idx_to_token"]),
    )
    print(
        "mappings (list) from index to caption token: ",
        data_dict["vocab"]["idx_to_token"],
    )
    print(
        "mappings (dict) from caption token to index: ",
        data_dict["vocab"]["token_to_idx"],
    )

    return data_dict


def get_toy_data(path: str = "final_data.json"):
    with open(path) as file:
        return json.load(file)


def train_captioner(
    model,
    image_data,
    caption_data,
    num_epochs,
    batch_size,
    learning_rate,
    lr_decay=1,
    device: torch.device = torch.device("cpu"),
):
    """Run optimization to train the image-captioning model."""
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), learning_rate
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: lr_decay**epoch
    )

    # Keep the final partial minibatch instead of silently discarding it.
    iter_per_epoch = math.ceil(image_data.shape[0] / batch_size)
    loss_history = []

    for i in range(num_epochs):
        start_t = time.time()
        for j in range(iter_per_epoch):
            images, captions = (
                image_data[j * batch_size : (j + 1) * batch_size],
                caption_data[j * batch_size : (j + 1) * batch_size],
            )
            images = images.to(device)
            captions = captions.to(device)

            loss = model(images, captions)
            optimizer.zero_grad()
            loss.backward()
            loss_history.append(loss.item())
            optimizer.step()
        end_t = time.time()

        print(
            "(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s".format(
                i + 1, num_epochs, loss.item(), end_t - start_t
            )
        )
        lr_scheduler.step()

    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training loss history")
    plt.show()
    return model, loss_history


def decode_captions(captions, idx_to_word):
    """
    Decode caption indices into words.

    Args:
        captions: Caption indices in a tensor of shape (N, T).
        idx_to_word: Mapping from vocabulary indices to words.

    Returns:
        A sentence, or a list of N sentences.
    """
    singleton = captions.ndim == 1
    captions = captions[None] if singleton else captions

    decoded = []
    n, t_max = captions.shape
    for i in range(n):
        words = []
        for t in range(t_max):
            token_idx = int(captions[i, t].item())
            word = idx_to_word[token_idx]
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words))

    if singleton:
        return decoded[0]
    return decoded


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_func,
    num_epochs,
    batch_size=32,
    warmup_lr=6e-6,
    warmup_interval=1000,
    lr=6e-4,
    device=torch.device("cpu"),
):
    """Train the Transformer used by the arithmetic-expression exercise."""
    print("Training started...")
    model = model.to(device)

    initial_lr = lr if warmup_interval is None else warmup_lr
    optimizer = torch.optim.Adam(
        model.parameters(), lr=initial_lr, betas=(0.9, 0.995), eps=1e-9
    )

    iteration = 0
    for epoch_num in range(num_epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        model.train()

        for inp, inp_pos, out, out_pos in train_dataloader:
            inp = inp.to(device)
            inp_pos = inp_pos.to(device)
            out = out.to(device)
            out_pos = out_pos.to(device)

            gnd = out[:, 1:].contiguous().view(-1).long()
            optimizer.zero_grad()

            pred = model(inp.long(), inp_pos, out.long(), out_pos)
            loss = loss_func(pred, gnd)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_tokens += gnd.numel()
            iteration += 1

            if warmup_interval is not None and iteration == warmup_interval:
                print(
                    f"End of warmup. Swapping learning rates from "
                    f"{warmup_lr} to {lr}"
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

        train_loss = epoch_loss / epoch_tokens
        val_loss, val_acc = val(
            model,
            val_dataloader,
            loss_func,
            batch_size=batch_size,
            device=device,
        )
        print(
            f"[epoch: {epoch_num + 1}] "
            f"[loss: {train_loss:.4f}] "
            f"val_loss: [val_loss {val_loss:.4f}] "
            f"val_acc: {val_acc:.4f}"
        )

    return model


def val(model, dataloader, loss_func, batch_size, device=torch.device("cpu")):
    """
    Evaluate token-level loss and token-level accuracy.

    ``batch_size`` is retained for compatibility with the original notebook;
    normalization uses the actual number of target tokens so partial batches are
    handled correctly.
    """
    del batch_size
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    num_correct = 0
    total = 0

    with torch.no_grad():
        for inp, inp_pos, out, out_pos in dataloader:
            inp = inp.to(device)
            inp_pos = inp_pos.to(device)
            out = out.to(device)
            out_pos = out_pos.to(device)

            gnd = out[:, 1:].contiguous().view(-1).long()
            pred = model(inp.long(), inp_pos, out.long(), out_pos)
            loss = loss_func(pred, gnd)

            pred_max = pred.argmax(dim=1)
            num_correct += pred_max.eq(gnd).sum().item()
            total += gnd.numel()
            total_loss += loss.item()

    if total == 0:
        raise ValueError("The validation dataloader produced no target tokens.")

    return total_loss / total, num_correct / total


def _get_subsequent_mask(seq):
    """Return a causal mask where True entries are hidden from attention."""
    n, k = seq.shape
    return torch.triu(
        torch.ones((n, k, k), dtype=torch.bool, device=seq.device), diagonal=1
    )


def inference(
    model,
    inp_exp,
    inp_exp_pos,
    out_pos_exp,
    out_seq_len,
    bos_idx=14,
):
    """
    Autoregressively generate an output sequence.

    ``bos_idx`` defaults to 14 to remain compatible with the original A5
    vocabulary. It can be passed explicitly as ``token_dict["BOS"]``.
    """
    model.eval()
    device = next(model.parameters()).device

    inp_exp = inp_exp.to(device)
    inp_exp_pos = inp_exp_pos.to(device)
    out_pos_exp = out_pos_exp.to(device)

    if inp_exp.ndim == 1:
        inp_exp = inp_exp.unsqueeze(0)
    if inp_exp_pos.ndim == 2:
        inp_exp_pos = inp_exp_pos.unsqueeze(0)
    if out_pos_exp.ndim == 2:
        out_pos_exp = out_pos_exp.unsqueeze(0)

    batch_size = inp_exp.shape[0]
    generated = torch.full(
        (batch_size, 1), bos_idx, dtype=torch.long, device=device
    )

    with torch.no_grad():
        ques_emb = model.emb_layer(inp_exp.long())
        q_emb_inp = ques_emb + inp_exp_pos
        enc_out = model.encoder(q_emb_inp)

        for _ in range(out_seq_len - 1):
            ans_emb = model.emb_layer(generated)
            ans_pos = out_pos_exp[:, : generated.shape[1], :]
            a_emb_inp = ans_emb + ans_pos

            # Match the causal masking used by Transformer.forward during training.
            mask = _get_subsequent_mask(generated)
            dec_out = model.decoder(a_emb_inp, enc_out, mask)
            next_word = dec_out[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_word], dim=1)

    return generated, model


def draw(data, x, y, ax):
    seaborn.heatmap(
        data,
        xticklabels=x,
        square=True,
        yticklabels=y,
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        ax=ax,
    )
