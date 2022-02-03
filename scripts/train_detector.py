
import argparse
import os

import torch
import torch.nn as nn

from src import deep

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='checkpoints/detector')
    parser.add_argument('--data-root', default='./data/human')
    parser.add_argument('--ssl-weights')

    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=0.03, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--rel-milestones', default=[0.6, 0.8], type=float, nargs='+')

    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--log-file', default='log.log')
    parser.add_argument('--log-interval', default=10, type=int)
    return parser.parse_args()

def main():
    args = get_args()

    if not os.path.exists(args.folder):
        os.makedirs(args.folder)

    # data
    train, val, test = deep.data.get_dataset(args.data_root, args.batch_size)
    # model
    model = deep.model.get_model(len(train.dataset.class_names), args.ssl_weights)
    # optimizer
    optimizer, scheduler = deep.model.get_optimizer(
        model, args.lr, args.momentum, args.epochs*len(train), args.rel_milestones)

    accelerator = deep.accelerate.MiniAccelerator(args.amp)
    train, val, test, model, optimizer = accelerator.prepare(
        train, val, test, model, optimizer)

    # status
    status = deep.status.Status(
        args.epochs*len(train), False,
        os.path.join(args.folder, args.log_file), args.log_interval)
    status.log_training(args, model)
    status.initialize_collector('Loss/train', 'Loss/val', 'Accuracy/train', 'Accuracy/val', 'LR')
    # loss
    criterion = nn.CrossEntropyLoss()
    # best model
    best_loss = 1e10

    while not status.is_end():
        model.train()
        for images, labels in train:
            with accelerator.autocast():
                output = model(images)
                loss = criterion(output, labels)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            accelerator.update()
            scheduler.step()

            _, prediction = output.max(1)
            correct = (prediction == labels).sum().item()

            status.update(**{
                'Loss/train': loss.item(),
                'Accuracy/train': correct/len(images.size(0)),
                'LR': optimizer.param_groups[0]['lr']})

        model.eval()
        loss    = 0
        correct = 0
        with torch.no_grad():
            for images, labels in val:
                with accelerator.autocast():
                    output = model(images)
                    batch_loss = criterion(output, labels)
                loss += batch_loss.item() * images.size(0)
                _, prediction = output.max(1)
                correct += (prediction == labels).sum().item()
        epoch_status = {'Loss/val': loss/len(val.dataset), 'Accuracy/val': correct/len(val.dataset)}
        status.update_collector(**epoch_status)
        status.log(f'[VALIDATION] STEP: {status.batches_done} / {status.max_iters} INFO: {epoch_status}')

        if epoch_status['Loss/val'] < best_loss:
            best_loss = epoch_status['Loss/val']
            best_model = model.state_dict()

    status.plot_loss(os.path.join(args.folder, 'loss.png'))

    model.load_state_dict(best_model)
    model.eval()
    loss    = 0
    correct = 0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for image, label in test:
            output = model(image)
            batch_loss = criterion(output, label)
            loss += batch_loss.item()
            _, pred = output.max(1)
            correct += (pred == label).sum().item()
            true_labels.append(label.item())
            pred_labels.append(pred.item())
    status.log(f'TEST\tLoss: {loss/len(test.dataset)}\tAccuracy: {correct/len(test.dataset)}')
    deep.status.log_test(
        status.log, true_labels, pred_labels,
        test.dataset.class_names, os.path.join(args.folder, 'cm.png'))
    torch.save(best_model, os.path.join(args.folder, 'model_final.pth'))

if __name__=='__main__':
    main()
