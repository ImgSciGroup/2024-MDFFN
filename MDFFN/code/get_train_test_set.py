import torch
import data_preprocess
from get_dataset import get_dataset as load_dataset


def prepare_train_test_data(config):
    # Load dataset configuration
    dataset_name = config['current_dataset']
    training_sample_count = config['train_set_num']
    patch_size = config['patch_size']

    image_pre_event, image_post_event, ground_truth = load_dataset(dataset_name)

    # Convert numpy arrays to torch tensors
    image_pre_event = torch.from_numpy(image_pre_event)
    image_post_event = torch.from_numpy(image_post_event)
    ground_truth = torch.from_numpy(ground_truth)

    # Check for NaN or Inf values in the tensors
    if torch.isnan(image_pre_event).any() or torch.isinf(image_pre_event).any():
        print("Pre-event image tensor contains NaN or Inf values.")
    if torch.isnan(image_post_event).any() or torch.isinf(image_post_event).any():
        print("Post-event image tensor contains NaN or Inf values.")
    if torch.isnan(ground_truth).any() or torch.isinf(ground_truth).any():
        print("Ground truth tensor contains NaN or Inf values.")

    # Permute dimensions to match CxHxW format
    image_pre_event = image_pre_event.permute(2, 0, 1)  # Channels first (CxHxW)
    image_post_event = image_post_event.permute(2, 0, 1)
    ground_truth_image = ground_truth

    # Construct samples
    padded_pre_event, padded_post_event, patch_coords = data_preprocess.construct_sample(
        image_pre_event, image_post_event, patch_size
    )

    # Divide samples into training and test sets
    data_samples = data_preprocess.select_sample(ground_truth_image, training_sample_count)

    # Store preprocessed data in dictionary
    data_samples['img1_pad'] = padded_pre_event
    data_samples['img2_pad'] = padded_post_event

    data_samples['patch_coordinates'] = patch_coords
    data_samples['img_gt'] = ground_truth_image
    data_samples['ori_gt'] = ground_truth

    return data_samples
