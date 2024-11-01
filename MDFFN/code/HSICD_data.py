import torch.utils.data as data


class HSICD_data(data.Dataset):
    def __init__(self, sample_data, config):

        self.model = config['phase']
        self.image_t1 = sample_data['img1_pad']
        self.image_t2 = sample_data['img2_pad']
        self.patch_coords = sample_data['patch_coordinates']
        self.ground_truth = sample_data['img_gt']

        if self.model == 'train':
            self.sample_indices = sample_data['train_sample_center']
        elif self.model == 'test':
            self.sample_indices = sample_data['test_sample_center']

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        sample_index = self.sample_indices[idx]
        coord_index = self.patch_coords[sample_index[0]]

        # Extract samples from padded images based on coordinates
        patch_image_t1 = self.image_t1[:, coord_index[0]:coord_index[1], coord_index[2]:coord_index[3]]
        patch_image_t2 = self.image_t2[:, coord_index[0]:coord_index[1], coord_index[2]:coord_index[3]]
        label = self.ground_truth[sample_index[1], sample_index[2]]

        return patch_image_t1, patch_image_t2, label, sample_index



