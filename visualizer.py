import numpy as np
import matplotlib.pyplot as plt
from conv_numpy import conv2d
from data_loader import load_single_image


def create_filter_bank():
    """
    Возвращает список 3D фильтров (ядра), каждый размером (3, 3, 3).
    Один фильтр = один канал на выходе.
    """
    # Фильтр вертикальных границ
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    # Горизонтальный фильтр
    sobel_y = np.array([[1,  2,  1],
                        [0,  0,  0],
                        [-1, -2, -1]])

    # Лапласиан
    laplace = np.array([[0,  1, 0],
                        [1, -4, 1],
                        [0,  1, 0]])

    # Размытие (блюр)
    blur = np.ones((3, 3)) / 9.0

    def to_3d(kernel2d):
        return np.stack([kernel2d] * 3, axis=-1)

    return {
        "Sobel X": to_3d(sobel_x),
        "Sobel Y": to_3d(sobel_y),
        "Laplace": to_3d(laplace),
        "Blur": to_3d(blur),
    }


def apply_filters(image, filters: dict, stride=1, padding=1):
    """
    Применяет все фильтры к изображению, возвращает словарь feature maps.
    """
    outputs = {}
    for name, kernel in filters.items():
        fmap = conv2d(image, kernel, stride=stride, padding=padding)
        outputs[name] = fmap
    return outputs


def plot_feature_maps(feature_maps: dict):
    """
    Визуализирует feature maps в одной строке.
    """
    n = len(feature_maps)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axs = [axs]
    for ax, (name, fmap) in zip(axs, feature_maps.items()):
        ax.imshow(fmap, cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image, _, _ = load_single_image(index=3)
    filters = create_filter_bank()
    fmap_dict = apply_filters(image, filters)
    plot_feature_maps(fmap_dict)
