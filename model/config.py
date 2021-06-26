
common_config = {
    'data_dir': './data/mjsynth/mnt/ramdisk/max/90kDICT32px/',
    'img_width': 100,
    'img_height': 32,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
}

train_config = {
    'epochs': 250,
    'train_batch_size': 128,
    'eval_batch_size': 512,
    'lr': 0.0005,
    'valid_interval': 25,
    'save_interval': 50,
    'cpu_workers': 4,
    'reload_checkpoint': None,
    'decode_method': 'greedy',
    'beam_size': 10,
    'weights_dir': 'weights/'
}
train_config.update(common_config)

evaluate_config = {
    'eval_batch_size': 512,
    'cpu_workers': 4,
    'reload_checkpoint': None,
    'decode_method': 'greedy',
    'beam_size': 10,
}
evaluate_config.update(common_config)
