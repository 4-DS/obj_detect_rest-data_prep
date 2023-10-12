import numpy as np


def cut_image_in_center(image_shape: tuple, max_slice_size, overlap_percent: float) -> np.ndarray:
    if type(max_slice_size) in [list, tuple, np.ndarray]:
        max_slice_size_x = max_slice_size[0]
        max_slice_size_y = max_slice_size[1]
    else:
        max_slice_size_x = max_slice_size
        max_slice_size_y = max_slice_size

    overlap = int(overlap_percent * max([max_slice_size_x, max_slice_size_y]))

    x_center_left = image_shape[1] // 2 - max_slice_size_x // 2

    count_left = int(np.ceil(x_center_left / max_slice_size_x))

    y_center_top = image_shape[0] // 2 - max_slice_size_y // 2

    count_top = int(np.ceil(y_center_top / max_slice_size_y))

    x_row_list = [x_center_left - max_slice_size_x*i + i *
                  overlap//2 for i in range(-count_left-1, count_left + 1)]
    y_row_list = [y_center_top - max_slice_size_y*j + j *
                  overlap//2 for j in range(-count_top-1, count_top + 1)]

    xy_array = np.array([[(x, y) for y in y_row_list]
                        for x in x_row_list]).reshape(-1, 2)
    x2y2_array = xy_array + [max_slice_size_x, max_slice_size_y]
    x1y1_array = xy_array.copy()

    x1y1_array[x1y1_array < 0] = 0

    x2y2_array[:, 0][x2y2_array[:, 0] > image_shape[1]] = image_shape[1]
    x2y2_array[:, 1][x2y2_array[:, 1] > image_shape[0]] = image_shape[0]

    w_h = x2y2_array - x1y1_array
    bad_slices = np.any(w_h <= overlap, axis=1)

    return np.hstack([x1y1_array, x2y2_array])[~bad_slices]


def cut_by_max_size(image_shape: tuple, max_slice_size: int, slice_overlap: float) -> np.ndarray:
    ofs = (max_slice_size * (1 - slice_overlap))

    x_row_list = []
    for i in range(int(np.ceil(image_shape[1] / ofs))):
        x_row_list.append(ofs * i)

    y_row_list = []
    for i in range(int(np.ceil(image_shape[0] / ofs))):
        y_row_list.append(ofs * i)

    xy_array = np.array([[(x, y) for y in y_row_list]
                        for x in x_row_list]).reshape(-1, 2)
    x2y2_array = xy_array + [max_slice_size, max_slice_size]
    x1y1_array = xy_array.copy()

    x1y1_array[x1y1_array < 0] = 0

    x2y2_array[:, 0][x2y2_array[:, 0] > image_shape[1]] = image_shape[1]
    x2y2_array[:, 1][x2y2_array[:, 1] > image_shape[0]] = image_shape[0]

    w_h = x2y2_array - x1y1_array
    bad_slices = np.any(w_h <= max_slice_size / 2, axis=1)
    return np.hstack([x1y1_array, x2y2_array])[~bad_slices].astype(np.int32)


if __name__ == "__main__":
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    mask = np.zeros((1200, 1900), dtype=np.uint8)
    xyxy = cut_image_in_center(mask.shape, 512, 0.28).tolist()

    for x1, y1, x2, y2 in xyxy:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, 2)

    plt.imshow(mask)
    plt.show()

    mask = np.zeros((1200, 1900), dtype=np.uint8)
    xyxy = cut_by_max_size(mask.shape, 512, 0.28).tolist()

    for x1, y1, x2, y2 in xyxy:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, 2)

    plt.imshow(mask)
    plt.show()
