import torch


def train_one_epoch(model, criterion, optimizer, train_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        total += targets.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()

    return train_loss/(batch_idx +1), correct/total


def evaluate(model, criterion, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
    return test_loss/(batch_idx+1), correct/total

