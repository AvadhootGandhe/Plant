import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

def train_resnet18():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder("data/train/", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)   # 3 classes example
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        for images, labels in train_loader:
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} Loss: {loss.item()}")

    torch.save(model.state_dict(), "models/resnet18.pt")