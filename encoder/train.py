
import torch
from torch.nn.parallel import DistributedDataParallel
import shutil
import os



class Trainer: 
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler, train_loader,
                 criterion, checkpoint_dir: str,
                 best_model_dir: str, args) -> None:
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_loader
        self.args = args
        self.checkpoint_dir = checkpoint_dir
        self.best_model_dir = best_model_dir
        self.criterion = criterion
        self.epochss_run = 0
        self.best_loss = 1e8
        self.model = model
        if os.path.exists(self.checkpoint_dir):
            print("Loading snapshot")
            self._load_snapshot()

    
    def _load_snapshot(self):
        snapshot = torch.load(self.checkpoint_dir)
        self.model.load_state_dict(snapshot['model_state_dict'])
        self.optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        self.epochss_run = snapshot['epochs']
        self.best_loss = snapshot['best_loss']

    def _save_snapshot(self, epochs, best_loss):
        snapshot = {
            'epochs': epochs,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': best_loss
        }
        torch.save(snapshot, self.checkpoint_dir)
        print(f"Epochs {epochs} | Training Snapshot saved at {self.checkpoint_dir} | Best loss is {best_loss}")

    def _run_batch(self, inputs):
        self.optimizer.zero_grad()
        outputs, _ = self.model(inputs)
        loss = self.criterion(outputs, inputs)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    
    def _run_epochs(self, epochs):
        print(f"[GPU {self.args.rank}] Epochs {epochs}, BatchSize: {self.args.batch_size} | Steps: {len(self.train_data)}")
        total_loss = 0
        for idx, batch in enumerate(self.train_data):  
            inputs = batch.cuda(self.args.gpu)
            loss = self._run_batch(inputs)
            total_loss += loss
            print(f"[GPU {self.args.rank}] Step/Steps: {idx}/{len(self.train_data)} | Total Loss: {total_loss/(idx + 1.0)} | Loss: {loss}")

        return total_loss

    def train(self):
        for epoch in range(self.epochss_run, self.args.epochs):
            self.model.train()
            total_loss = self._run_epochs(epoch)
            average_loss = total_loss / len(self.train_data)
            if (epoch + 1) % self.args.checkpoint_interval == 0:
                can_replace_best = False
                if average_loss < self.best_loss:
                    if self.args.rank == 0:
                        print(f"Epochs is {epoch} and new best loss is {average_loss} and old was {self.best_loss}")  # noqa: E501
                    can_replace_best = True
                    self.best_loss = average_loss
                if self.args.rank == 0:
                    self._save_snapshot(epoch, self.best_loss)
                    if can_replace_best:
                        shutil.copy(self.checkpoint_dir, self.best_model_dir)

        