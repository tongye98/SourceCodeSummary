import torch 

if __name__ == "__main__":

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=100)

    lambda1 = lambda epoch: 0.6 ** epoch 
    print(type(lambda1))
    print(optimizer.param_groups)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    lrs = []
    for i in range(10):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    
    for idx, item in enumerate(lrs):
        print("{} lr = {}".format(idx, round(item,8)))