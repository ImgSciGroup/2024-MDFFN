import torch

def label_to_image(predicted_labels, ground_truth_image):
    predicted_image = torch.zeros_like(ground_truth_image)
    num_samples = predicted_labels.shape[0]

    for i in range(num_samples):
        x_coord = int(predicted_labels[i][1])
        y_coord = int(predicted_labels[i][2])
        label = predicted_labels[i][3]
        predicted_image[x_coord][y_coord] = label

    return predicted_image
