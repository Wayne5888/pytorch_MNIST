#import dependency
import torch
from PIL import Image
import numpy as np
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#GetData
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)

#classes 0-10

#Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64,64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6) * (28-6), 10)
        )

    def forward(self, x):
        return self.model(x)

# Instance of the neural network, loss, optimizer

clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    

    #Training function
    for epoch in range(10):
        for batch in dataset:
            X, y = batch
            X, y = X.to('cuda'), y.to('cuda')
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            #Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch : {epoch} loss is {loss.item()}")
    with open("mode_state.pt", "wb") as f:
        save(clf.state_dict(), f)

    with open('mode_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    img = Image.open('img_3.JPG')
    gray_image = img.convert('L')
    image_tensor = ToTensor()(gray_image).unsqueeze(0).to('cuda')
    print(gray_image.getbands()) 
    print(torch.argmax(clf(image_tensor)))

