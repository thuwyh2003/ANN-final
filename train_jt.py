# coding=utf-8

from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time
import jittor
from datetime import timedelta

#import torch
#import torch.distributed as dist

from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
# from jittor import vis
# import jittorvis as vis
#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP

from models.modeling_jt import VisionTransformer, CONFIGS
from utils.scheduler_jt import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils_jt import get_loader
from utils.dist_util_jt import get_world_size
from tensorboardX import SummaryWriter       #wyh
# from jittor import logger
logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def simple_accuracy(preds, labels):
    correct = (preds == labels)

    return correct.mean()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    #dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    # jittor.distributed.all_reduce(rt, op=jittor.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    if args.fp16:
        checkpoint = {
            'model': model_to_save.state_dict(),
            #'amp': amp.state_dict()
            'amp': jittor.float16
        }
    else:
        checkpoint = {
            'model': model_to_save.state_dict(),
        }
    #torch.save(checkpoint, model_checkpoint)
    jittor.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,smoothing_value=args.smoothing_value)
    #print(model)
    model.load_from(np.load(args.pretrained_dir))
    #print(model)
    if args.pretrained_model is not None:
        '''pretrained_model = torch.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)'''
        pretrained_model = jittor.load(args.pretrained_model)['model']
        model.load_parameters(pretrained_model)

        
    # model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    jittor.set_global_seed(args.seed)
    # if args.n_gpu > 0:                     #暂时取消，用单卡跑
    #     jittor.set_global_seed(args.seed)

def valid(args, model, writer, test_loader, global_step):
    # Validation!
    jittor.flags.use_cuda=1
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    #loss_fct = torch.nn.CrossEntropyLoss()
    loss_fct = jittor.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch

        x=x.squeeze(1)
        #with torch.no_grad():
        with jittor.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            #preds = torch.argmax(logits, dim=-1)
            preds,_ = jittor.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    #accuracy = torch.tensor(accuracy).to(args.device)
    accuracy = jittor.array(accuracy).to(args.device)

    #dist.barrier()
    #jittor.distributed.barrier()
    #val_accuracy = reduce_mean(accuracy, args.nprocs)
    val_accuracy = accuracy.detach().cpu().numpy()

    logger.info("\n")
    print("Validation Results")
    print("Global Steps: %d" % global_step)
    print("Valid Loss: %2.5f" % eval_losses.avg)
    print("Valid Accuracy: %2.5f" % val_accuracy)
    if args.local_rank in [-1, 0]:
        writer.add_scalar("test/accuracy", scalar_value=val_accuracy, global_step=global_step)
        
    return val_accuracy

def train(args, model):
    """ Train the model """
    jittor.flags.use_cuda=1
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = jittor.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # if args.fp16:
    #     '''model, optimizer = amp.initialize(models=model,
    #                                       optimizers=optimizer,
    #                                       opt_level=args.fp16_opt_level)
    #     amp._amp_state.loss_scalers[0]._loss_scale = 2**20'''
    #     jittor.flags.use_fp16 = True                                 #AttributeError: 'jittor_core.Flags' object has no attribute 'use_fp16'

    # Distributed training
    if args.local_rank != -1:
        #model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
        model = jittor.nn.DataParallel(model)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    jittor.get_device_count() if jittor.has_cuda else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

     #      有问题
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    start_time = time.time()
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch

            x = x.squeeze(1)

            loss, logits = model(x, y)
            loss = loss.mean()

            #preds = torch.argmax(logits, dim=-1)
            
            preds,_ = jittor.argmax(logits, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            '''if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()'''

            optimizer.backward(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                '''if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    '''
                optimizer.clip_grad_norm(args.max_grad_norm)
                
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0:
                    #with torch.no_grad():
                    with jittor.no_grad():
                        accuracy = valid(args, model, writer, test_loader, global_step)
                    if args.local_rank in [-1, 0]:
                        if best_acc < accuracy:
                            save_model(args, model)
                            best_acc = accuracy
                        logger.info("best accuracy so far: %f" % best_acc)
                    model.train()

                if global_step % t_total == 0:
                    break
        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        #accuracy = torch.tensor(accuracy).to(args.device)
        accuracy = jittor.array(accuracy).to(args.device)
        #dist.barrier()
        # jittor.distributed.barrier()
        # train_accuracy = reduce_mean(accuracy, 1)
        train_accuracy = accuracy.detach().cpu().numpy()
        print("train accuracy so far: %f" % train_accuracy)
        losses.reset()
        if global_step % t_total == 0:
            break

    writer.close()
    print("Best Accuracy: \t%f" % best_acc)
    print("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))

def main():
    #torch.cuda.empty_cache()      #WYH
    # torch.cuda.set_per_process_memory_fraction(0.9, device=torch.device(f'cuda:{local_rank}'))
    jittor.clean()
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017"], default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='./')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/opt/tiger/minist/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    # parser.add_argument("--local_rank", type=int, default=-1,
    #                     help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    # parser.add_argument('--local-rank',type=int,default=0,help="local rank for distributed training")  #WYH
    args = parser.parse_args()

    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU & distributed training
    args.local_rank = int(os.environ.get('LOCAL_RANK', -1))     #WYH   不需要手动传输local_rank参数了。
    #print("local_rank",args.local_rank)
    '''
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        print(args.n_gpu)
        breakpoint()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()'''
    
    jittor.flags.use_cuda=1
    # from datetime import timedelta

    # if args.local_rank == -1:
    #     if jittor.backends.cuda.is_available():
    #         jittor.flags.use_cuda = 1 
    #         args.n_gpu = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    #     else:
    #         jittor.flags.use_cuda = 0
    #         args.n_gpu = 1
    #     print(f"Number of GPUs available: {args.n_gpu}")
    #     breakpoint()
    # else:
    #     jittor.flags.use_cuda = 1
    #     #device = jittor.core.Device(jittor.core.CUDA, args.local_rank)
    #     # jittor.distributed.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
    #     args.n_gpu = 1
    args.device = 'cuda'
    # args.nprocs =len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))      #暂时取消，用单卡跑


    # Setup logging
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',     #暂时取消，用单卡跑
    #                     datefmt='%m/%d/%Y %H:%M:%S',
    #                     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # logger.warning("Process rank: %s,  n_gpu: %s, distributed training: %s, 16-bits training: %s" %(args.local_rank,  args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)
    # Training
    train(args, model)

if __name__ == "__main__":
    main()
