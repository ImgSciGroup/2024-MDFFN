# current_dataset = 'Wetland_new'
current_dataset = 'Hermiston'
current_model = '_mine'
current = current_dataset + current_model

# 0.Super parameter
patch_size = 9
lr = 5e-3
bs_number = 32
epoch_number = 40


# 1. data
phase = ['train', 'test', 'no_gt']
train_set_num = 0.02

data = dict(
    current_dataset=current_dataset,
    train_set_num=train_set_num,
    patch_size=patch_size,

    train_data=dict(
        phase=phase[0]
    ),
    test_data=dict(
        phase=phase[1]
    ),
)

# 2. model
model = dict(
    in_fea_num=242  # Farmland 155 Hermiston 242 BayArea224    132
)

# 3. train
train = dict(
    optimizer=dict(
        typename='SGD',
        lr=lr,
        momentum=0.9,
        weight_decay=5e-3
    ),
    train_model=dict(
        gpu_train=True,
        gpu_num=1,
        workers_num=12,
        epoch=epoch_number,
        batch_size=bs_number,
        lr=lr,
        lr_adjust=True,
        lr_gamma=0.1,
        lr_step=[35, 70],
        save_folder='./weights/' + current_dataset + '/',
        save_name=current,
        reuse_model=False,
        reuse_file='./weights/' + current + '_Final.pth',
    )
)

# 4. test
test = dict(
    batch_size=1000,
    gpu_train=True,
    gpu_num=1,
    workers_num=8,
    model_weights='./weights/' + current_dataset + '/' + current + '_Final.pth',
    save_name=current,
    save_folder='../result' + '/' + current_dataset
)



