from lightning.pytorch.callbacks import Callback
from datetime import datetime
import os

class SaveLoRAWeightsCallback(Callback):
    def __init__(self, root_dir="lora_weights"):
        self.root_dir = root_dir
        self.run_dir = None

    def on_fit_start(self, trainer, pl_module):
        if self.run_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_dir = os.path.join(self.root_dir, timestamp)

    def on_train_epoch_end(self, trainer, pl_module):

        if hasattr(pl_module, "hparams") and "lora_enabled" in pl_module.hparams:
            lora_enabled = pl_module.hparams.lora_enabled

        if not lora_enabled:
            return

        model = getattr(pl_module, "network", getattr(pl_module, "model", None))
        if not hasattr(model, "save_pretrained"):
            return
        
        save_path = os.path.join(
            self.run_dir, 
            f"epoch-{trainer.current_epoch:02d}"
        )
        
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)