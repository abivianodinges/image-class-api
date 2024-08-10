def mainMethod(filePath):
    import torchvision
    import torch
    import torchvision.transforms as transforms
    import PIL.Image as Image

    classes = [
        "Mantled Howler",
        "Patas Monkey",
        "Bald Ukari",
        "Japanese Macaque",
        "Pygmy Marmoset",
        "White Headed Capuchin",
        "Silvery Marmoset",
        "Common Squirrel Monkey",
        "Black Headed Night Monkey",
        "Nilgiri Langur"
    ]

    #example filepath: test/macaque_4.jpg

    
    model = torch.load('best_model.pth')
    mean = [0.4363, 0.4328, 0.3291]
    std = [0.2129, 0.2075, 0.2038]

    image_transforms = transforms.Compose([
        transforms.Resize((220,220)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    def classify(model, image_transforms, image_path, classes):
        model = model.eval()
        image = Image.open('test/' + image_path)
        image = image_transforms(image).float()
        image = image.unsqueeze(0)

        output = model(image)
        _, predicted = torch.max(output.data, 1)

        print(classes[predicted.item()])
        print('filepath ' + filePath)
        return(classes[predicted.item()])
    return classify(model, image_transforms, filePath, classes)
