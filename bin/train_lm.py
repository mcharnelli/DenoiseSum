import argparse
from DenoiseSum import DATA_PATH
from DenoiseSum.LanguageModel.training import train_language_model








if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset', default='rotten', type=str)
  parser.add_argument('--no_instance', default=40, type=int)

  parser.add_argument('--batch_size', default=16, type=int)
  parser.add_argument('--word_dim', default=300, type=int)
  parser.add_argument('--hidden_dim', default=512, type=int)
  parser.add_argument('--disc_size', default=50, type=int)

  parser.add_argument('--num_epoch', default=20, type=int)
  parser.add_argument('--eval_every', default=2500, type=int)
  parser.add_argument('--stop_after', default=5, type=int)

  parser.add_argument('--train_file', default='reviews_large_train.csv', type=str)
  parser.add_argument('--dev_file', default='reviews_large_validation.csv', type=str)
  parser.add_argument('--test_file', default='reviews_large_test.csv', type=str)

  parser.add_argument('--replace_word_dict', default='False', type=str)
  parser.add_argument('--retrain_model', default='False', type=str)

  parser.add_argument('--model_file', default='lm.model', type=str)
  parser.add_argument('--sos', default=2, type=int)
  parser.add_argument('--eos', default=3, type=int)

  parser.add_argument('--coverage_rate', default=0, type=int)

  args = parser.parse_args()
  args.train_file = DATA_PATH / args.dataset / args.train_file
  args.dev_file =  DATA_PATH / args.dataset / args.dev_file
  args.test_file = DATA_PATH / args.dataset / args.test_file

  if args.replace_word_dict == 'False':
    args.replace_word_dict = False
  else:
     args.replace_word_dict = True

  if args.retrain_model == 'False':
    args.retrain_model = False
  else:
     args.retrain_model = True

  (DATA_PATH / 'model' / args.dataset).mkdir(parents = True, exist_ok=True)
  args.model_file = DATA_PATH / 'model' / args.dataset / args.model_file 

  train_language_model(args)
