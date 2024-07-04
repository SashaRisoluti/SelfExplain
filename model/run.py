import lightning as pl
import pytorch_lightning
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.core import LightningDataModule
import random
import numpy as np
import logging
from argparse import ArgumentParser
import resource
from data import ClassificationData
from SE_XLNet import SEXLNet

def get_train_steps(dm):
  total_devices = args.num_gpus * 5 #args.num_nodes non è stato inserito il numero di nodi quindi metto 5 per vedere se il resto del codice funziona
  train_batches = len(dm.train_dataloader()) // total_devices
  """if args.accumulate_grad_batches is None:
    return (args.max_epochs * train_batches)
  else:
   return args.max_epochs * train_batches // args.accumulate_grad_batches"""
  return (args.max_epochs * train_batches)




rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
# init: important to make sure every node initializes the same weights
SEED = 18
np.random.seed(SEED)
random.seed(SEED)
pl.fabric.utilities.seed.seed_everything(SEED)


# argparser
parser = ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--clip_grad', type=float, default=1.0)
parser.add_argument("--dataset_basedir", help="Base directory where the dataset is located.", type=str)
parser.add_argument("--concept_store", help="Concept store file", type=str)
parser.add_argument("--model_name", default='xlnet-base-cased', type=str, help="Model to use.")
parser.add_argument("--gamma", default=0.01, type=float, help="Gamma parameter")
parser.add_argument("--lamda", default=0.01, type=float, help="Lamda Parameter")
parser.add_argument("--topk", default=100, type=int,help="Topk GIL concepts")
#parser.add_argument('--min_lr', type=float, default=1e-5)
#parser.add_argument('--h_dim', type=int, default=256)
#parser.add_argument('--n_heads', type=int, default=8)
#parser.add_argument('--kqv_dim', type=int, default=64)
#parser.add_argument('--num_classes', type=int, default=10)
#parser.add_argument('--lr', type=float, default=0.001)
#parser.add_argument('--weight_decay', type=float, default=0.0001)
#parser.add_argument('--warmup_prop', type=float, default=0.1)
parser.add_argument('--max_epochs', type=int, default=10)
#parser.add_argument('--gpus', type=int, default=1)
#parser.add_argument('--accelerator', type=str, default='ddp')


parser = SEXLNet.add_model_specific_args(parser)

args = parser.parse_args()
# print(args)
args.num_gpus = len(str(args.num_gpus).split(","))


logging.basicConfig(level=logging.INFO)

# Step 1: Init Data
logging.info("Loading the data module")
dm = ClassificationData(basedir=args.dataset_basedir, tokenizer_name=args.model_name, batch_size=args.batch_size)

# Step 2: Init Model
logging.info("Initializing the model")
model = SEXLNet(hparams=args)
model.hparams.warmup_steps = int(get_train_steps(dm) * model.hparams.warmup_prop)
lr_monitor = LearningRateMonitor(logging_interval='step')

# Step 3: Start
logging.info("Starting the training")
checkpoint_callback = ModelCheckpoint(
    filename='{epoch}-{step}-{val_acc_epoch:.4f}',
    save_top_k=3,
    verbose=True,
    monitor='val_acc_epoch',
    mode='max'
)

trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], val_check_interval=0.5, gradient_clip_val=args.clip_grad, track_grad_norm=2)
trainer.fit(model, dm)
# trainer.test()
