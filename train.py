import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss
from tqdm import tqdm
from dataset import Synth90kDataset, synth90k_collate_fn
from model import CRNN
from evaluate import evaluate
from model.config import train_config as config


def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, 2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths) / batch_size

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    eval_batch_size = config['eval_batch_size']
    lr = config['lr']
    show_interval = config['show_interval']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']
    valid_max_iter = config['valid_max_iter']

    img_width = config['img_width']
    img_height = config['img_height']
    data_dir = config['data_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_dataset = Synth90kDataset(root_dir=data_dir, mode='train',
                                    img_height=img_height, img_width=img_width)
    valid_dataset = Synth90kDataset(root_dir=data_dir, mode='dev',
                                    img_height=img_height, img_width=img_width)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=synth90k_collate_fn)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=eval_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=synth90k_collate_fn)

    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'])
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    assert save_interval % valid_interval == 0
    i = 1
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        with tqdm(train_loader, unit="batch") as tepoch:
            for train_data in tepoch:
                loss = train_batch(crnn, train_data, optimizer, criterion, device)
                tepoch.set_postfix(loss=loss)
        if epoch % valid_interval == 0:
            evaluation = evaluate(crnn, valid_loader, criterion,
                                  decode_method=config['decode_method'],
                                  beam_size=config['beam_size'])
            print('valid_evaluation: loss={loss}, acc={acc}'.format(**evaluation))

            if epoch % save_interval == 0:
                prefix = 'persian-lpr'
                loss = evaluation['loss']
                save_model_path = os.path.join(config['weights_dir'],
                                               f'{prefix}_loss_{loss}.pt')
                torch.save(crnn.state_dict(), save_model_path)
                print('save model at ', save_model_path)

if __name__ == '__main__':
    main()
