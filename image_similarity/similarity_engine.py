import torch

def train_epoch(encoder,decoder, optimizer, loss_fn, data_loader, device):
    encoder.train()
    decoder.train()
    total_loss = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        encoder_x = encoder(x)
        output = decoder(encoder_x)
        loss = loss_fn(output, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evl_epoch(encoder,decoder, loss_fn, data_loader, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            encoder_x = encoder(x)
            output = decoder(encoder_x)
            loss = loss_fn(output, y)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def create_embedding(encoder,full_loder,device):
    encoder.eval()
    embeddings = torch.empty(0)
    with torch.no_grad():
        for x, y in full_loder:
            x = x.to(device)
            encoder_x = encoder(x).cpu()
            embeddings = torch.cat((embeddings, encoder_x), 0)
    return embeddings