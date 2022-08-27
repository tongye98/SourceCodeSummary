from torch.utils.tensorboard import SummaryWriter 

tb = SummaryWriter(log_dir="models/test/tensorboard3")

tb.add_scalar("Train/batch_loss", 4, 1)
tb.add_scalar("Train/batch_loss", 1, 2)
tb.close()