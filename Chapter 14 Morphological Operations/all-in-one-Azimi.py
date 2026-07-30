import cv2
import numpy as np
import os
import shutil
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import matplotlib.pyplot as plt
import json
import pandas as pd


def multi_scale_morphological_decomposition(image, structuring_elements):
    decomposed_images = []
    opened_images = []
    current_image = image.copy()
    for selem in structuring_elements:
        opened_image = cv2.morphologyEx(current_image, cv2.MORPH_OPEN, selem)
        opened_images.append(opened_image)
        current_image = current_image - opened_image
        decomposed_images.append(current_image)
    return decomposed_images, opened_images


def structuring_elements(shape, size, count, angle):
    # Convert the angle from radians to degrees for OpenCV rotation
    angle_deg = angle * 180 / np.pi
    selem = []
    size_param = [3, size]
    top, bottom, left, right = size_param[1] // 2, size_param[1] // 2, size_param[0] // 2, size_param[0] // 2
    selem.append(np.array([[1]]))
    if shape == 'RECT':
        temp = cv2.getStructuringElement(cv2.MORPH_RECT, (size_param[0], size_param[1]))
    elif shape == 'CROSS':
        temp = cv2.getStructuringElement(cv2.MORPH_CROSS, (size_param[0], size_param[1]))
    elif shape == 'ELLIPSE':
        temp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size_param[0], size_param[1]))
    else:
        raise ValueError(f"Unknown shape {shape}")
    temp = cv2.copyMakeBorder(temp, max(size_param), max(size_param), max(size_param), max(size_param),
                              cv2.BORDER_CONSTANT, value=0)
    h, w = temp.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1)
    temp = cv2.warpAffine(temp.astype(np.uint8), rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
    selem.append(temp)
    for c in range(count - 1):
        padded_image = cv2.copyMakeBorder(selem[-1], top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        dilated_image = cv2.dilate(padded_image, temp, iterations=1)
        selem.append(dilated_image)
    selem.reverse()
    return selem


def run_htd(img, folder, shapes, sizes, counts, angles):
    images = []
    labels = []
    # sequence through hyperparameters provided by the user
    for count in counts:
        for shape in shapes:
            for size in sizes:
                for angle in angles:
                    # get structuring elements by converting the angle to radians
                    selem = structuring_elements(shape, size, count,
                                                 angle * np.pi / (max(angles) if max(angles) != 0 else 1))
                    # save images of structuring elements
                    n = 0
                    se_folder = os.path.join(folder, "structuring_elements")
                    for s in selem:
                        cv2.imwrite(f"{se_folder}/structuring_element_{shape}_{size}_{count}_{angle}_{n}.jpg", s * 255)
                        n += 1
                    # HTD: get decomposed and texture images using the multi-scale morphological decomposition
                    d_images, o_images = multi_scale_morphological_decomposition(img, selem)
                    n = 0
                    for d, o in zip(d_images, o_images):
                        decomposed_folder = os.path.join(folder, "decomposed_images")
                        texture_folder = os.path.join(folder, "texture_images")
                        cv2.imwrite(f"{decomposed_folder}/decomposed_image_{shape}_{size}_{count}_{angle}_{n}.jpg", d)
                        cv2.imwrite(f"{texture_folder}/texture_image_{shape}_{size}_{count}_{angle}_{n}.jpg", o)
                        labels.append([shape, size, count, angle, n])
                        images.append(o)
                        n += 1
    return images, labels


def reshape_data(img):
    # Convert a collection of N 2D images into a 3D array (i x j x N)
    reshaped_img = []
    for j in range(len(img[0])):
        temp1 = []
        for k in range(len(img[0][0])):
            temp2 = []
            for i in range(len(img)):
                temp2.append(img[i][j][k])
            temp1.append(temp2)
        reshaped_img.append(temp1)
    reshaped_img = np.array(reshaped_img)
    return reshaped_img


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(os.path.join(folder_path, "structuring_elements"))
        os.makedirs(os.path.join(folder_path, "decomposed_images"))
        os.makedirs(os.path.join(folder_path, "texture_images"))
        os.makedirs(os.path.join(folder_path, "logistic_results"))
    else:
        # If it does exist, delete all its contents
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        os.makedirs(os.path.join(folder_path, "structuring_elements"))
        os.makedirs(os.path.join(folder_path, "decomposed_images"))
        os.makedirs(os.path.join(folder_path, "texture_images"))
        os.makedirs(os.path.join(folder_path, "logistic_results"))


def get_grid(img):
    # approximate regions where each texture is to be found
    L = list(img.shape)
    num_clusters = 6
    x1 = L[0] // 2
    xR = [[0, x1], [0, x1], [0, x1], [x1, L[0]], [x1, L[0]], [x1, L[0]]]
    y1 = L[1] // 3
    yR = [[0, y1], [y1, 2 * y1], [2 * y1, L[1]], [0, y1], [y1, 2 * y1], [2 * y1, L[1]]]
    region_x = []
    region_y = []
    for i in range(6):
        x = xR[i][0] + 30
        y = yR[i][0] + 30
        h = int(x1 * 2 / 3)
        w = int(y1 * 2 / 3)
        region_x.append([x, x + h])
        region_y.append([y, y + w])
    return xR, yR, num_clusters, region_x, region_y


def get_data_subset(vec, xR, yR):
    x = []
    y = []
    for k in range(len(xR)):
        for i in range(xR[k][0], xR[k][1]):
            for j in range(yR[k][0], yR[k][1]):
                x.append(vec[i][j])
                y.append(k)
    x = np.array(x)
    return x, y


def eval_logistic(model, inputs, outputs, num_clusters, reshaped, xR, yR):
    outputs = np.array(outputs)
    y_new_pred = model.predict(inputs)
    y_new_proba = model.predict_proba(inputs)
    accuracy = []
    for i in range(num_clusters):
        mask = (outputs == i)
        temp_y = outputs[mask]
        cnt_y = np.sum(mask)
        cnt_x = (y_new_pred[mask] == i).sum()
        accuracy.append([i, cnt_x, cnt_y, cnt_x / cnt_y if cnt_y != 0 else 0])
    cm = confusion_matrix(outputs, y_new_pred)
    reshaped_labels = np.zeros((len(reshaped), len(reshaped[0])))
    reshaped_prob = np.zeros((len(reshaped), len(reshaped[0]), num_clusters))
    n = 0
    for k in range(len(xR)):
        for i in range(xR[k][0], xR[k][1]):
            for j in range(yR[k][0], yR[k][1]):
                reshaped_labels[i, j] = y_new_pred[n]
                reshaped_prob[i, j, :] = y_new_proba[n]
                n += 1
    return accuracy, reshaped_labels, reshaped_prob, cm


def overlay_red_mask(img, mask, alpha=0.5):
    gray_image_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    red_mask = np.zeros_like(gray_image_bgr)
    red_mask[mask] = [0, 0, 255]
    blended_image = cv2.addWeighted(gray_image_bgr, 1.0, red_mask, alpha, 0)
    return blended_image


def overlay_color_mask(isgray, img, mask, index, alpha=0.5):
    if isgray:
        gray_image_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        gray_image_bgr = img.copy()

    colors = [
        [0, 0, 255],  # Red
        [0, 255, 0],  # Green
        [255, 0, 0],  # Blue
        [0, 255, 255],  # Yellow
        [255, 255, 0],  # Cyan
        [255, 0, 255],  # Magenta
        [0, 165, 255],  # Orange
        [128, 0, 128],  # Purple
        [203, 192, 255],  # Pink
        [42, 42, 165]  # Brown
    ]
    color_mask = np.zeros_like(gray_image_bgr)
    color_mask[mask] = colors[index % len(colors)]
    blended_image = cv2.addWeighted(gray_image_bgr, 1.0, color_mask, alpha, 0)
    return blended_image


def run_logistic_training_and_eval(img, vec, file_path):
    xR, yR, num_clusters, region_x, region_y = get_grid(img)
    x, y = get_data_subset(vec, region_x, region_y)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(max_iter=10000)
    model.fit(x_scaled, y)
    coefficients = np.array(model.coef_)
    x_full, y_full = get_data_subset(vec, xR, yR)
    x_scaled_full = scaler.transform(x_full)
    accuracy, logistic_labels, logistic_prob, cm = eval_logistic(model, x_scaled_full, y_full, num_clusters, vec, xR,
                                                                 yR)
    img_out = []
    for i in range(num_clusters):
        masked_img = overlay_red_mask(img, logistic_labels == i)
        img_out.append(masked_img)
    img_prob = []
    for i in range(num_clusters):
        temp = []
        for j in range(len(logistic_prob)):
            temp2 = []
            for k in range(len(logistic_prob[j])):
                temp2.append(int(logistic_prob[j][k][i] * 255))
            temp.append(temp2)
        temp = np.array(temp)
        img_prob.append(temp)
    for i in range(num_clusters):
        if i == 0:
            new_img = overlay_color_mask(True, img, logistic_labels == i, i, alpha=0.5)
        else:
            new_img = overlay_color_mask(False, new_img, logistic_labels == i, i, alpha=0.5)
    return coefficients, accuracy, cm, img_out, img_prob, new_img, model, scaler


if __name__ == "__main__":
    # Prompt the user for parameter inputs
    shapes_input = input("Enter kernel shapes separated by comma (e.g. RECT,CROSS,ELLIPSE): ")
    shapes = [s.strip() for s in shapes_input.split(",") if s.strip()]

    sizes_input = input("Enter kernel sizes separated by comma(e.g. 3,7,11,15), where the resulting kernel will be 3 X size: ")
    try:
        sizes = [int(s.strip()) for s in sizes_input.split(",") if s.strip()]
    except ValueError:
        print("Invalid sizes entered. Using default sizes [3,7,11,15].")
        sizes = [3, 7, 11, 15]

    counts_input = input("Enter the number of decompositions separated by comma (e.g. 8,16,32): ")
    try:
        counts = [int(s.strip()) for s in counts_input.split(",") if s.strip()]
    except ValueError:
        print("Invalid counts entered. Using default counts [8,16,32].")
        counts = [8, 16, 32]

    angle_input = input("Enter the number of angular increments for the kernel (e.g. 8 for angles 0,1,...,7): ")
    try:
        angle_count = int(angle_input.strip())
    except ValueError:
        print("Invalid angle count entered. Using default of 8 increments.")
        angle_count = 8
    angles = list(range(angle_count))

    # List of example image labels; update or modify as necessary
    fabric_labels = ['fabric1', 'fabric2', 'fabric3']

    for label in fabric_labels:
        # read image in grayscale and prepare directory
        img = cv2.imread(f"img/{label}.jpg", 0)
        if img is None:
            print(f"Image for {label} not found. Skipping...")
            continue
        htd_folder = f"{label}/htd"
        create_folder(htd_folder)
        # run HTD using user provided parameters
        images, labels_list = run_htd(img, htd_folder, shapes, sizes, counts, angles)
        unique_combinations = {tuple(sub_list[:4]) for sub_list in labels_list}
        unique_combinations_list = list(unique_combinations)
        perf = None
        coef = None

        for u in unique_combinations_list:
            logistic_folder = os.path.join("logistic_results", f"shape-{u[0]}_size-{u[1]}_levels-{u[2]}_angle-{u[3]}")
            full_logistic_folder = os.path.join(htd_folder, logistic_folder)
            os.makedirs(full_logistic_folder, exist_ok=True)
            indices = [i for i, row in enumerate(labels_list) if tuple(row[:4]) == u]
            image_subsets = [images[i] for i in indices]
            images2 = reshape_data(image_subsets)
            coefficients, accuracy, cm, img_out, img_prob, new_img, mdl, scl = run_logistic_training_and_eval(img,
                                                                                                              images2,
                                                                                                              htd_folder)
            for i in range(len(img_out)):
                file_name = os.path.join(full_logistic_folder, 'class_' + str(i) + '.jpg')
                cv2.imwrite(file_name, img_out[i])
            for i in range(len(img_prob)):
                file_name = os.path.join(full_logistic_folder, 'class_prob_' + str(i) + '.jpg')
                cv2.imwrite(file_name, img_prob[i])
            file_name = os.path.join(full_logistic_folder, 'full.jpg')
            cv2.imwrite(file_name, new_img)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mdl.classes_)
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(full_logistic_folder, "confusion_matrix.jpg"), format="jpeg", dpi=300)
            plt.close()
            with open(os.path.join(full_logistic_folder, "model.pkl"), "wb") as file_out:
                pickle.dump(mdl, file_out)
            with open(os.path.join(full_logistic_folder, "scale.pkl"), "wb") as file_out:
                pickle.dump(scl, file_out)
            df = pd.DataFrame(np.transpose(coefficients))
            df['shape'] = u[0]
            df['size'] = u[1]
            df['levels'] = u[2]
            df['angle'] = u[3]
            df['decomposition'] = range(u[2] + 1)
            if coef is None:
                coef = df.copy()
            else:
                coef = pd.concat([coef, df], axis=0)
            perf_hdr = ['class', 'count predicted', 'count expected', 'proportion correct']
            if perf is None:
                perf = pd.DataFrame(accuracy, columns=perf_hdr)
                perf['shape'] = u[0]
                perf['size'] = u[1]
                perf['levels'] = u[2]
                perf['angle'] = u[3]
            else:
                temp = pd.DataFrame(accuracy, columns=perf_hdr)
                temp['shape'] = u[0]
                temp['size'] = u[1]
                temp['levels'] = u[2]
                temp['angle'] = u[3]
                perf = pd.concat([perf, temp], axis=0)
        coef.to_excel(os.path.join(htd_folder, "logistic_results", "coef.xlsx"), index=False)
        perf.to_excel(os.path.join(htd_folder, "logistic_results", "perf.xlsx"), index=False)