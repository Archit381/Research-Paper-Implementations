import logging
from rich.logging import RichHandler
import torch
import torchvision
import torchvision.transforms as transforms


logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

log = logging.getLogger("rich")

def generate_mnist_dataset(batch_size = 128):
    log.info("Generating MNIST Sequence Modeling Dataset")

    SEQ_LENGTH = 784
    N_CLASSSES =  256
    IN_DIM = 1

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x.view(IN_DIM, SEQ_LENGTH).t()*255).int()
            )
        ]
    )

    try: 
        train = torchvision.datasets.MNIST(
            "./data", train=True, download = True, transform = tf
        )

        test = torchvision.datasets.MNIST(
            "./data", train=False, download = True, transform = tf
        )
    
    except Exception as e:
        log.error("[bold red blink]Downloading Dataset Failed![/]", extra={"markup": True})
        raise Exception(e)

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size = batch_size,
        shuffle = True
    )

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size = batch_size,
        shuffle = False
    )

    return train_loader, test_loader, N_CLASSSES, SEQ_LENGTH, IN_DIM

