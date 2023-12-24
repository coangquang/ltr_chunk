import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .model import CrossEncoder

class CrossEncoderCELoss(object):
    def calc(self,
             logits,
             labels):
        loss = F.cross_entropy(logits,
                               labels)
        max_score, max_idxs = torch.max(logits, 1)
        correct_predictions_count = (max_idxs == labels).sum()
        return loss, correct_predictions_count
    
class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))
    
class CrossTrainer():
    """
    Trainer for cross-encoder
    """
    def __init__(self,
                 args,
                 train_loader, 
                 val_loader,
                 test_loader):
        
        self.stop = False

        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parallel = True if torch.cuda.device_count() > 1 else False
        print("No of GPU(s):",torch.cuda.device_count())
        
        self.model = CrossEncoder(model_checkpoint = self.args.cross_checkpoint,
                                  representation = self.args.cross_representation,
                                  dropout = self.args.cross_dropout)
        if self.args.cross_load_path:
            self.model.load_state_dict(torch.load(self.args.cross_load_path))
        if self.parallel:
            self.model = DataParallel(self.model)
        self.model.to(self.device)
        self.criterion = CrossEncoderCELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=args.cross_lr) 
        self.scheduler = WarmupLinearSchedule(self.optimizer, int(0.1 * len(self.train_loader) * self.args.cross_num_epochs), len(self.train_loader) * self.args.cross_num_epochs)
        
        self.epoch = 0
        self.patience_counter = 0
        self.best_val_acc = 0.0
        self.steps_count = []
        self.valid_losses = []
        self.valid_acc = []
        self.steps = 0
        
    def train_crossencoder(self):
        # Compute loss and accuracy before starting (or resuming) training.
        print("\n",
              20 * "=",
              "Validation before training",
              20 * "=")
        val_time, val_loss, val_acc = self.validate()
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(val_time, val_loss, (val_acc*100)))
        print("\n",
              20 * "=",
              "Training cross-encoder model on device: {}".format(self.device),
              20 * "=")
        while self.epoch < self.args.cross_num_epochs:
            self.epoch +=1
            print("* Training epoch {}:".format(self.epoch))
            epoch_avg_loss, epoch_accuracy, epoch_time = self.train()
            print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
                  .format(epoch_time, epoch_avg_loss, (epoch_accuracy*100)))
            if self.stop:
                break
        
        if self.parallel:   
            torch.save(self.model.module.state_dict(), self.args.cross_final_path) 
        else:
            torch.save(self.model.state_dict(), self.args.cross_final_path)    
        # Plotting of the loss curves for the train and validation sets.
        plt.figure()
        plt.plot(self.steps_count, self.valid_losses, "-r")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend(["Validation loss"])
        plt.title("Cross entropy loss")
        plt.show()
    
        plt.figure()
        #plt.plot(self.steps_count, self.train_acc, '-r')
        plt.plot(self.steps_count, self.valid_acc, "-b")
        plt.xlabel("step")
        plt.ylabel("accuracy")
        plt.legend(["Validation accuracy"])
        plt.title("Accuracy")
        plt.show()
        
        print("* Testing with final model:")
        test_loss, test_acc = self.test()
        print("-> Accuracy: {:.4f}%".format((test_acc*100)))
        #best_path = "/kaggle/working/" + self.args.cross_best_path
        if self.parallel:
            self.model.module.load_state_dict(torch.load(self.args.cross_best_path))
        else:
            self.model.load_state_dict(torch.load(self.args.cross_best_path))
        print("* Testing with best model:")
        test_loss, test_acc = self.test()
        print("-> Accuracy: {:.4f}%".format((test_acc*100)))
        if self.parallel:
            return self.model.module
        else:
            return self.model
    
    def train(self):
        self.model.train()
        epoch_start = time.time()
        batch_time_avg = 0.0
        epoch_loss = 0.0
        epoch_correct = 0
        tqdm_batch_iterator = tqdm(self.train_loader)
        for i, batch in enumerate(tqdm_batch_iterator):
            self.steps += 1
            
            batch_start = time.time()
            loss, num_correct = self.step(batch)
            batch_time_avg += time.time() - batch_start
            epoch_loss += loss
            epoch_correct += num_correct

            description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                        .format(batch_time_avg/(i+1),
                        epoch_loss/(i+1))
            tqdm_batch_iterator.set_description(description)
            
            if self.steps % self.args.cross_eval_steps == 0:
                self.steps_count.append(self.steps)
                print("\t* Validation for step {}:".format(self.steps))
                epoch_time, epoch_loss, epoch_accuracy = self.validate()
                self.valid_losses.append(epoch_loss)
                self.valid_acc.append(epoch_accuracy.to('cpu')*100)
                print("\t-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
                      .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
                
                if epoch_accuracy <= self.best_val_acc:
                    self.patience_counter += 1

                else:
                    self.best_val_acc = epoch_accuracy
                    self.patience_counter = 0
                    if self.parallel:
                        torch.save(self.model.module.state_dict(), self.args.cross_best_path)
                    else:
                        torch.save(self.model.state_dict(), self.args.cross_best_path)
            
                if self.patience_counter >= self.args.cross_patience:
                    self.stop = True
                    print("-> Early stopping: patience limit reached, stopping...")
                    break

        epoch_time = time.time() - epoch_start
        epoch_avg_loss = epoch_loss / len(self.train_loader)
        epoch_accuracy = epoch_correct / len(self.train_loader.dataset)

        return epoch_avg_loss, epoch_accuracy, epoch_time
    
    def step(self, batch):
        self.model.train()
        input_ids, attn_mask, labels = tuple(t.to(self.device) for t in batch)

        self.optimizer.zero_grad()
        logits = self.model(input_ids, attn_mask)
        loss, num_correct = self.criterion.calc(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), num_correct
    
    def validate(self):
        self.model.eval()

        epoch_start = time.time()
        total_loss = 0.0
        total_correct = 0
        accuracy = 0

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                input_ids, attn_mask, labels = tuple(t.to(self.device) for t in batch)
                logits = self.model(input_ids, attn_mask)
                loss, num_correct = self.criterion.calc(logits, labels)
                total_loss += loss.item()
                total_correct += num_correct

            epoch_time = time.time() - epoch_start
            val_loss = total_loss/len(self.val_loader)
            accuracy = total_correct/len(self.val_loader.dataset)

        return epoch_time, val_loss, accuracy
    
    def test(self):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        with torch.no_grad():
            tqdm_batch_iterator = tqdm(self.test_loader)
            for i, batch in enumerate(tqdm_batch_iterator):
                input_ids, attn_mask, labels = tuple(t.to(self.device) for t in batch)
                logits = self.model(input_ids, attn_mask)
                loss, num_correct = self.criterion.calc(logits, labels)
                total_loss += loss.item()
                total_correct += num_correct
                test_loss = total_loss/len(self.test_loader)
                accuracy = total_correct/len(self.test_loader.dataset)
        return test_loss, accuracy