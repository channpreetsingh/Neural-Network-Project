import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Grayscale

# Get data
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, batch_size=32, shuffle=True)

# Image classifier neural network class
class ImageClassifier(nn.Module): # nn is like a base class for all pytorch modelss
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(  #it defines model in sequential manner like each layer is connected to the next layer
            nn.Conv2d(1, 32, (3, 3)),  # 32 feature maps, 3x3 kernel
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),  
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),  
            nn.ReLU(),
            nn.Flatten(),
            
            nn.Linear(64 * 22 * 22, 10) 
        )

    def forward(self, x):
        return self.model(x)

# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cpu')
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(clf.parameters(), lr=1e-3) # lr is learning rate which is like the size of steps taken by opt to minimize the loss or errosrs

# Training loop
if __name__ == "__main__":
    # Load pre-trained model
    with open(r"C:/Users/chanp/OneDrive/Documents/SEM3/AI and ML/Projects and Practise (Python)/Neural network/model.pth", "rb") as f:
        clf.load_state_dict(load(f))
    
    # Load and preprocess the image
    img = Image.open(r"C:\Users\chanp\OneDrive\Documents\SEM3\AI and ML\Projects and Practise (Python)\Neural network\2.jpg")
    
    # Resize the image to (28, 28) and convert it to grayscale
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to match MNIST input size (28x28)
    
    # Convert the image to a tensor
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
    
    # Print the predicted label
    with torch.no_grad(): #Disables gradient calculation as we r not training the model. 
        prediction = torch.argmax(clf(img_tensor))
        print(f"Predicted Label: {prediction.item()}")





   
#     for epoch in range(10):  # Train for 10 epochs
#         total_loss = 0
#         for batch in dataset:
#             X, y = batch
#             X, y = X.to('cpu'), y.to('cpu')
            
#             optimizer.zero_grad()
#             output = clf(X)
#             l = loss_fn(output, y)
            
#             # Backpropagation
#             l.backward()
#             optimizer.step()
            
#             total_loss += l.item()
        
#         print(f"Epoch {epoch+1} finished with loss {total_loss / len(dataset)}")

# # Save model
# with open("model.pth", "wb") as f:
#     save(clf.state_dict(), f)