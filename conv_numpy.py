import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_single_image, show_image


def pad_image(image: np.ndarray, pad: int) -> np.ndarray:
    """
    Паддит изображение нулями по краям (zero padding).
    image: np.array (H, W, C)
    pad: int — число пикселей для паддинга
    """
    return np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')


def conv2d(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Применяет 2D-свёртку вручную.

    image: np.ndarray (H, W, C) — изображение (например, 32x32x3)
    kernel: np.ndarray (kH, kW, C) — ядро фильтра (например, 3x3x3)
    stride: int — шаг
    padding: int — паддинг

    return: feature map (out_H, out_W)
    """
    assert image.shape[2] == kernel.shape[2], "Channel mismatch!"

    image_padded = pad_image(image, padding)
    H, W, C = image_padded.shape
    kH, kW, _ = kernel.shape

    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    output = np.zeros((out_H, out_W))

    for y in range(out_H):
        for x in range(out_W):
            region = image_padded[y*stride:y*stride+kH, x*stride:x*stride+kW, :]
            output[y, x] = np.sum(region * kernel)

    return output


if __name__ == "__main__":
    

    image, _, _ = load_single_image(index=3)

    # Простой edge detection фильтр (один и тот же на каждый канал)
    edge_kernel = np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]])

    # Преобразуем в форму (kH, kW, C)
    kernel_3d = np.stack([edge_kernel] * 3, axis=-1)

    feature_map = conv2d(image, kernel_3d, stride=1, padding=1)

    plt.imshow(feature_map, cmap='gray')
    plt.title("Feature Map")
    plt.axis('off')
    plt.show()
