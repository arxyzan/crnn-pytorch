import argparse
import torch
import numpy as np
from PIL import Image

from model.config import common_config as config
from dataset import Synth90kDataset
from model import CRNN
from model.ctc_decoder import ctc_decode

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', type=str,
                    help='path to the image', default='./novel.jpg')
parser.add_argument('-m', '--model', type=str,
                    help='path to the model weights', default='./weights/crnn.pt')
parser.add_argument('--decoder', type=str,
                    help='ctc decoder, (greedy,beam_search,prefix_beam_search', default='greedy')
parser.add_argument('--beam_size', type=int, help='beam size', default=10)
parser.add_argument('--device', type=str, help='gpu or cpu', default='gpu')


def load_image(image_path, img_width=100, img_height=32):
    image = Image.open(image_path).convert('L')

    image = image.resize((img_width, img_height), resample=Image.BILINEAR)
    image = np.array(image)
    image = image.reshape((1, 1, img_height, img_width))
    image = (image / 127.5) - 1.0

    image = torch.FloatTensor(image)
    return image


def predict_single(crnn, image_path, label2char, decode_method, beam_size):
    crnn.eval()
    with torch.no_grad():
        device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

        image = load_image(image_path=image_path).to(device)

        logits = crnn(image)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                           label2char=label2char)
    return preds


if __name__ == '__main__':
    args = parser.parse_args()
    image_path = args.image
    model_path = args.model
    decoder = args.decoder
    beam_size = args.beam_size
    device = 'cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    img_height = config['img_height']

    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'])
    crnn.load_state_dict(torch.load(model_path, map_location=device))
    crnn.to(device)

    preds = predict_single(crnn,
                           image_path=image_path,
                           label2char=Synth90kDataset.LABEL2CHAR,
                           decode_method=decoder,
                           beam_size=beam_size)

    print(f"Predicted Text: {''.join(preds[0])}")
