import wandb


def log_losses_wandb_tracking(
    logwandb, num_batches, local_rank, losses, loss, val=False
):
    if val:
        val_ = " val"
    else:
        val_ = ""
    if logwandb and ((num_batches - 1) % 10) == 0 and local_rank == 0:
        wandb.log(
            {
                "lv + lbeta" + val_ : loss.item(),
                "loss" + val_ + " lv": losses[0].item(),
                "loss" + val_ + " beta": losses[1].item(),
                "loss" + val_ + " beta sig": losses[4].item(),
                "loss" + val_ + " beta noise": losses[5].item(),
                "loss" + val_ + " attractive": losses[2].item(),
                "loss" + val_ + " repulsive": losses[3].item(),
            }
        )
