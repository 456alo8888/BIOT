import os
import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pyhealth.metrics import multiclass_metrics_fn

from datasets.nfeeg_dataset import N_LoadDataset , N_CustomDataset
from datasets.uet175_dataset import UET_LoadDataset , UETCustomDataset
import datasets.eegtals_dataset
# from finetune_trainer import Trainer

from pytorch_lightning.loggers import WandbLogger
import wandb

wandb.login(key = "9af519f93bfdf0c0ee9458c50cd05342021c9e71")



from model import (
    SPaRCNet,
    ContraWR,
    CNNTransformer,
    FFCL,
    STTransformer,
    BIOTClassifier,
)
from utils import TUEVLoader, HARLoader


class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model

    def training_step(self, batch, batch_idx):
        X, y = batch
        prod = self.model(X)
        loss = nn.CrossEntropyLoss()(prod, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = convScore.cpu().numpy()
            step_gt = y.cpu().numpy()
        return step_result, step_gt

    def validation_epoch_end(self, val_step_outputs):
        result = []
        gt = np.array([])
        for out in val_step_outputs:
            result.append(out[0])
            gt = np.append(gt, out[1])

        result = np.concatenate(result, axis=0)
        result = multiclass_metrics_fn(
            gt, result, metrics=["accuracy", "cohen_kappa", "f1_weighted"]
        )
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_cohen", result["cohen_kappa"], sync_dist=True)
        self.log("val_f1", result["f1_weighted"], sync_dist=True)
        print(result)

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = convScore.cpu().numpy()
            step_gt = y.cpu().numpy()
        return step_result, step_gt

    def test_epoch_end(self, test_step_outputs):
        result = []
        gt = np.array([])
        for out in test_step_outputs:
            result.append(out[0])
            gt = np.append(gt, out[1])

        result = np.concatenate(result, axis=0)
        result = multiclass_metrics_fn(
            gt, result, metrics=["accuracy", "cohen_kappa", "f1_weighted"]
        )
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_cohen", result["cohen_kappa"], sync_dist=True)
        self.log("test_f1", result["f1_weighted"], sync_dist=True)

        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return [optimizer]  # , [scheduler]


def prepare_TUEV_dataloader(args):
    # set random seed
    seed = 4523
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf"

    train_files = os.listdir(os.path.join(root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.1), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), train_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_eval"), test_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), val_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_HAR_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/HAR/processed/"

    train_files = os.listdir(os.path.join(root, "train"))
    test_files = os.listdir(os.path.join(root, "test"))
    val_files = os.listdir(os.path.join(root, "val"))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "train"),
                  train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader

def prepare_NFEEG_dataloader(args):
    load_dataset = N_LoadDataset(args)
    data_loader = load_dataset.get_data_loader()
    return data_loader

def prepare_UET175_dataloader(args):
    load_dataset = UET_LoadDataset(args)
    data_loader = load_dataset.get_data_loader()
    return data_loader

def prepare_EEGTALS_dataloader(args):
    load_dataset = datasets.eegtals_dataset.LoadDataset(args)
    data_loader = load_dataset.get_data_loader()
    return data_loader


def supervised(args):
    # get data loaders
    if args.dataset == "NMF":
        dataloader = prepare_NFEEG_dataloader(args)
        train_loader, test_loader, val_loader = dataloader['train'] , dataloader['test'] , dataloader['val']
    elif args.dataset == "UET175":
        dataloader = prepare_UET175_dataloader(args)
        train_loader, test_loader, val_loader = dataloader['train'] , dataloader['test'] , dataloader['val']
    elif args.dataset == "EEGTALS":
        dataloader = prepare_EEGTALS_dataloader(args)
        train_loader, test_loader, val_loader = dataloader['train'] , dataloader['test'] , dataloader['val']
    else:
        raise NotImplementedError


    # define the model
    if args.model == "SPaRCNet":
        model = SPaRCNet(
            in_channels=args.in_channels,
            sample_length=int(args.sample_length * args.sampling_rate),
            n_classes=args.n_classes,
            block_layers=4,
            growth_rate=16,
            bn_size=16,
            drop_rate=0.5,
            conv_bias=True,
            batch_norm=True,
        )

    elif args.model == "ContraWR":
        model = ContraWR(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
        )

    elif args.model == "CNNTransformer":
        model = CNNTransformer(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.sampling_rate,
            steps=args.hop_length // 5,
            dropout=0.2,
            nhead=4,
            emb_size=256,
            n_segments=4 if args.dataset == "HAR" else 5,
        )

    elif args.model == "FFCL":
        model = FFCL(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
            sample_length=int(args.sample_length * args.sampling_rate),
            shrink_steps=16 if args.dataset == "HAR" else 20,
        )

    elif args.model == "STTransformer":
        model = STTransformer(
            emb_size=256,
            depth=4,
            n_classes=args.n_classes,
            channel_legnth=int(
                args.sampling_rate * args.sample_length
            ),  # (sampling_rate * duration)
            n_channels=args.in_channels,
        )

    elif args.model == "BIOT":
        model = BIOTClassifier(
            n_classes=args.n_classes,
            # set the n_channels according to the pretrained model if necessary
            n_channels=args.in_channels,
            n_fft=args.token_size,
            hop_length=args.hop_length,
        )
        if args.pretrain_model_path and (args.sampling_rate == 200):

            ckpt = torch.load(args.pretrain_model_path, map_location="cpu")
            state_dict = ckpt["state_dict"]  # đây mới là weight thật

            # Nếu state_dict có prefix như "biot.", thì bỏ prefix đi
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("biot."):
                    new_state_dict[k.replace("biot.", "")] = v
                else:
                    new_state_dict[k] = v

            model.biot.load_state_dict(new_state_dict, strict=False)  # strict=False để bỏ qua mismatch
            # model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")

    else:
        raise NotImplementedError
    lightning_model = LitModel_finetune(args, model)

    # logger and callbacks

    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}"
    logger = TensorBoardLogger(
        save_dir="./",
        version=version,
        name="log",
    )
    # Khởi tạo logger
    exp_name = f"{args.dataset}"
    wandb_logger = WandbLogger(project="BIOT", name=exp_name)
    early_stop_callback = EarlyStopping(
        monitor="val_cohen", patience=30, verbose=False, mode="max"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_cohen",
        dirpath="./pretrain_models",
        filename="best-{epoch:02d}-{val_cohen:.4f}",
        save_top_k=1,
        mode="max",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,   # hoặc devices=[0] nếu bạn muốn chọn GPU id cụ thể
        strategy=DDPStrategy(find_unused_parameters=False),
        benchmark=True,
        enable_checkpointing=True,
        logger=[wandb_logger, logger],
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # t = Trainer(args, dataloader, model)
    # t.train_for_multiclass()

    

    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path="best", dataloaders=test_loader
    )[0]
    print(pretrain_result)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=32, help="number of workers")
    parser.add_argument("--dataset", type=str, default="TUAB", help="dataset")
    parser.add_argument(
        "--model", type=str, default="SPaRCNet", help="which supervised model to use"
    )
    parser.add_argument(
        "--in_channels", type=int, default=12, help="number of input channels"
    )
    parser.add_argument(
        "--sample_length", type=float, default=10, help="length (s) of sample"
    )
    parser.add_argument(
        "--n_classes", type=int, default=1, help="number of output classes"
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=200, help="sampling rate (r)"
    )
    parser.add_argument("--token_size", type=int,
                        default=200, help="token size (t)")
    parser.add_argument(
        "--hop_length", type=int, default=100, help="token hop length (t - p)"
    )
    parser.add_argument(
        "--pretrain_model_path", type=str, default="/mnt/disk1/aiotlab/hieupc/New_CBraMod/BIOT/pretrained-models/epoch=epoch=39_step=step=382000.ckpt", help="pretrained model path"
    )
    parser.add_argument('--datasets_dir', type=str,
                        default='/data/datasets/BigDownstream/Faced/processed',
                        help='datasets_dir')
    
    args = parser.parse_args()
    print(args)

    supervised(args)
