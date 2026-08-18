from praatio import tgio
import os
import fileinput
import argparse


parser = argparse.ArgumentParser(
    usage='Create phonemes by frames, phonemes transcription by frames, words by frames files'
)
parser.add_argument('--f', dest="fps", type=float, default=25.0, help='Frames-per-second in video')
parser.add_argument('--p', dest="path", type=str, default="/default", help="Path where to store files")
parser.add_argument('--t', dest="tg_path", type=str, required=True, help="Path to TextGrid input file")
parser.add_argument('--d', dest='phonemekeys_dict', type=str, required=True, help="Path to phonemes-keys dictionary")

args = parser.parse_args()

tg_path = args.tg_path

with fileinput.FileInput(tg_path, inplace=True, backup='.bak') as file:
    for line in file:
        print(line.replace('text = ""', 'text = "<silence:>"'), end='')

fps = args.fps
phonemekeys_dict = args.phonemekeys_dict
output_path = args.path

tg = tgio.openTextgrid(tg_path)
russian_words = tg.tierDict['ORT-MAU'].entryList
phonemes_words = tg.tierDict['KAN-MAU'].entryList
phonemes = tg.tierDict['MAU'].entryList

phonemes_frames_out = output_path + '/phonemes_frames.txt'
phonemekeys_frames_out = output_path + '/phonemekeys_frames.txt'
phonemeswords_frames_out = output_path + '/phonemeswords_frames.txt'
words_frames_out = output_path + '/words_frames.txt'


def map_unknown_label(label, convert_to_keys):
    if label.startswith('<') and label.endswith('>'):
        return 'p!' if convert_to_keys else label
    raise KeyError(label)


def make_phonemes_frames_file(output_file, input_dict, fps, phonemes_keys_file, convert_to_keys=True):
    key_to_phonemes = {}
    with open(phonemes_keys_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            if convert_to_keys:
                key_to_phonemes[items[0]] = items[1].split('\n')[0]
            else:
                if items[0] != '<p:>':
                    key_to_phonemes[items[0]] = items[0]
                else:
                    key_to_phonemes[items[0]] = 'p!'

    with open(output_file, 'w') as f:
        for el in input_dict:
            start_frame = int(el.start * fps)
            end_frame = int(el.end * fps)
            lab = el.label
            value = key_to_phonemes.get(lab)
            if value is None:
                value = map_unknown_label(lab, convert_to_keys)
                key_to_phonemes[lab] = value

            for i in range(start_frame, end_frame):
                if i != 0:
                    f.write('\n')
                f.write(value)


def make_words_frames_file(output_file, input_dict, fps):
    with open(output_file, 'w') as f:
        for el in input_dict:
            start_frame = int(el.start * fps)
            end_frame = int(el.end * fps)
            lab = el.label

            for i in range(start_frame, end_frame):
                if i != 0:
                    f.write('\n')
                f.write(lab)


if not os.path.exists(output_path):
    os.makedirs(output_path)

make_phonemes_frames_file(phonemes_frames_out, phonemes, fps, phonemekeys_dict, convert_to_keys=False)
make_phonemes_frames_file(phonemekeys_frames_out, phonemes, fps, phonemekeys_dict, convert_to_keys=True)
make_words_frames_file(phonemeswords_frames_out, phonemes_words, fps)
make_words_frames_file(words_frames_out, russian_words, fps)

print('Files successful create in %s' % (output_path))
