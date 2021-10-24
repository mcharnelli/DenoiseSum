from  argparse import ArgumentParser
from DenoiseSum import DATA_PATH

from DenoiseSum.training import train_language_model






if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('dataset', default='rotten', type=str)

  args = parser.parse_args()
  file_path = DATA_PATH / args.dataset / 'raw' / 'file.json'