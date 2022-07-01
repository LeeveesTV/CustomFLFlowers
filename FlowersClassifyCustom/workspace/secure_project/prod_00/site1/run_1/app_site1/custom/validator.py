
import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

from torchvision.utils import make_grid
from torchvision import models as models 
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor

from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from simple_network import SimpleNetwork
import os

class Validator(Executor):
    
    def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION):
        super(Validator, self).__init__()

        self._validate_task_name = validate_task_name

        # Setup the model
        self.model = SimpleNetwork()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        data_dir = os.getcwd() + "/flowers"
        print("data_dir", data_dir)
        transformer = torchvision.transforms.Compose(
            [ # Applying Augmentation
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.RandomRotation(40),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        
        database = ImageFolder(data_dir, transform=transformer)
        batch_size = 32
        validation_size = 200
        training_size = len(database) - validation_size
        
        train_ds, val_ds_main = random_split(database, [training_size, validation_size])
        val_ds, test_ds = random_split(val_ds_main, [100, 100])
        
        self.test_loader = DataLoader(val_ds, batch_size)
        #self.test_loader = DataLoader(self.test_data, batch_size=64, shuffle=True)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                # Get validation accuracy
                val_accuracy = self.do_validation(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(fl_ctx, f"Accuracy when validating {model_owner}'s model on"
                                      f" {fl_ctx.get_identity_name()}"f's data: {val_accuracy}')

                dxo = DXO(data_kind=DataKind.METRICS, data={'val_acc': val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def do_validation(self, weights, abort_signal):
        self.model.load_state_dict(weights)

        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                if abort_signal.triggered:
                    return 0

                images, labels = images.to(self.device), labels.to(self.device)
                images_v = Variable(images)
                output = self.model(images_v)

                _, pred_label = torch.max(output, 1)

                correct += (pred_label == labels).sum().item()
                total += images.size()[0]

            metric = correct/float(total)
            print(metric)
        return metric
