from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os, math, torch

@dataclass
class TrainingEvent:
    step: int
    epoch: int
    train_loss: float
    val_loss: float = None
    model: object = field(default=None, repr=False)

class TrainingObserver(ABC):
    @abstractmethod
    def on_event(self, event: TrainingEvent): ...

class TrainingSubject:
    def __init__(self):
        self._observers = []
    def attach(self, o): self._observers.append(o)
    def detach(self, o): self._observers.remove(o)
    def notify(self, event):
        for o in self._observers: o.on_event(event)

class ConsoleLogObserver(TrainingObserver):
    def on_event(self, event):
        val = f"  val_loss={event.val_loss:.4f}" if event.val_loss else ""
        print(f"[epoch {event.epoch} | step {event.step}]  train_loss={event.train_loss:.4f}{val}")

class CheckpointObserver(TrainingObserver):
    def __init__(self, ckpt_dir="checkpoints", save_every=200):
        self.ckpt_dir = ckpt_dir
        self.save_every = save_every
        os.makedirs(ckpt_dir, exist_ok=True)
    def on_event(self, event):
        if event.step % self.save_every != 0 or event.model is None: return
        path = os.path.join(self.ckpt_dir, f"ckpt_step_{event.step:06d}.pt")
        torch.save({"step": event.step, "model_state": event.model.state_dict()}, path)

class EarlyStoppingObserver(TrainingObserver):
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._best = math.inf
        self._bad = 0
    def on_event(self, event):
        if event.val_loss is None: return
        if event.val_loss < self._best - self.min_delta:
            self._best = event.val_loss
            self._bad = 0
        else:
            self._bad += 1
            if self._bad >= self.patience:
                raise StopIteration
