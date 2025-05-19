import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def load_single_image(index=0):
    """
    Загружает одно изображение из CIFAR-10 и возвращает его как NumPy-массив (H, W, C).
    """
    transform = transforms.ToTensor()

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    image, label = dataset[index]
    image_np = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)

    return image_np, label, dataset.classes[label]

def show_image(img_np, label=None):
    """
    Отображает изображение NumPy.
    """
    plt.imshow(img_np)
    if label:
        plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    image, label, label_name = load_single_image(index=10)
    print("Label:", label_name)
    show_image(image, label_name)
