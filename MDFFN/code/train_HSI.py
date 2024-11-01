
import os
import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim

import configs as cfg
from HSICD_data import HSICD_data
from assessment import accuracy_assessment
import train
from get_train_test_set import prepare_train_test_data as load_datasets
# tools
from show import label_to_image
import test_moudle

# models
# from CSANet import Finalmodel as main_model
from Net import Net as Model


def main():
    dataset_name = cfg.current_dataset
    model_version = cfg.current_model
    full_model_name = f"{dataset_name}{model_version}"
    print(f'Model: {full_model_name}')

    data_config = cfg.data
    model_config = cfg.model
    train_config = cfg.train['train_model']
    optim_config = cfg.train['optimizer']
    test_config = cfg.test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data and split into train/test sets
    datasets = load_datasets(data_config)
    ground_truth_img = datasets['img_gt']
    train_dataset = HSICD_data(datasets, data_config['train_data'])
    test_dataset = HSICD_data(datasets, data_config['test_data'])

    # Initialize model
    model = Model(model_config['in_fea_num']).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=optim_config['lr'],
                          momentum=optim_config['momentum'],
                          weight_decay=optim_config['weight_decay'])

    # Train model
    # train.train(train_dataset, model, loss_function, optimizer, device, train_config)

    # Test model
    train_predictions, train_accuracy = test_moudle.test(
        (train_dataset, datasets['img_gt'], model, device, test_config))
    test_predictions, test_accuracy = test_moudle.test((test_dataset, datasets['img_gt'], model, device, test_config))

    # Post-processing
    combined_predictions = torch.cat([train_predictions, test_predictions], dim=0)
    print(f'Training accuracy: {train_accuracy:.2f}%, Test accuracy: {test_accuracy:.2f}%')

    predicted_img = label_to_image(combined_predictions, ground_truth_img)
    conf_matrix, overall_accuracy, kappa, precision, recall, f1_score, class_accuracy = accuracy_assessment(
        ground_truth_img, predicted_img)

    assessment_results = [
        round(overall_accuracy, 4) * 100,
        round(kappa, 4),
        round(f1_score, 4) * 100,
        round(precision, 4) * 100,
        round(recall, 4) * 100,
        full_model_name
    ]
    print('Assessment results:', assessment_results)

    # Save results
    output_folder = test_config['save_folder']
    output_name = test_config['save_name']

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sio.savemat(os.path.join(output_folder, f"{output_name}.mat"), {
        "predict_img": np.array(predicted_img.cpu()),
        "oa": assessment_results
    })
    predicted_img = predicted_img.numpy()
    cv2.imwrite(os.path.join(output_folder, f"{output_name}+MAFE.bmp"), predicted_img * 255)
    print('Prediction image saved successfully!')


if __name__ == '__main__':
    main()
