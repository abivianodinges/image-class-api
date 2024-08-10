def mainMethod(filePath):
    import torchvision
    import torch
    import torchvision.transforms as transforms
    import PIL.Image as Image
    import torch.nn as nn
    import torch.nn.functional as F

    classes = [
        "Violence",
        "NonViolence"
    ]

    #example filepath: test/macaque_4.jpg

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 53 * 53, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, len(classes))

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    model = Net()

    model.load_state_dict(torch.load('violence_model.pt'))
    
    image_transforms = transforms.Compose([
        transforms.Resize(226),
        transforms.CenterCrop(224),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def classify(model, image_transforms, image_path, classes):
        model = model.eval()
        image = Image.open('violenceTest/' + image_path)
        image = image_transforms(image).float()
        image = image.unsqueeze(0)

        output = model(image)
        _, predicted = torch.max(output.data, 1)

        print(classes[predicted.item()])
        return(classes[predicted.item()])
    return classify(model, image_transforms, filePath, classes)
