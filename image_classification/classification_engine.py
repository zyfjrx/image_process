import torch

def train_epoch(model, optimizer, loss_fn, data_loader, device):
    model.train()
    total_loss = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_fn(output, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(data_loader)

def evl_epoch(model, loss_fn, data_loader, device):
    model.eval()
    total_loss = 0
    correct_num = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_num += pred.eq(y).sum()
    return total_loss / len(data_loader),correct_num