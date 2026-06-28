import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from parameter_config import TRAIN_CONFIG, USE_RELU
from utils import build_dense_model, DATA_DIR


DATASET_REGISTRY = {
    'mnist': {
        'class': datasets.MNIST,
        'mean': (0.1307,),
        'std': (0.3081,),
    },
    'fashion': {
        'class': datasets.FashionMNIST,
        'mean': (0.2860,),
        'std': (0.3530,),
    },
}


def get_loader(dataset_name, train, batch_size, device):
    ds_info = DATASET_REGISTRY[dataset_name]
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(ds_info['mean'], ds_info['std']),
    ])
    ds = ds_info['class'](root=DATA_DIR, train=train, download=True, transform=tf)
    pin = (device == 'cuda')
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=pin)


def train_model(model, train_loader, test_loader, num_epochs, lr, device, name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        print(f"[{name}] Epoch {epoch+1}/{num_epochs} — test acc: {correct/total*100:.2f}%")

    return model


def evaluate(model, loader, device, name):
    model.eval().to(device)
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    acc = correct / total * 100
    print(f"  {name}: {acc:.2f}%")
    return acc


def main():
    cfg = TRAIN_CONFIG
    use_relu = USE_RELU

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"ReLU: {'enabled' if use_relu else 'disabled'}")

    tgt = cfg['target_model']
    src = cfg['source_model']
    bs_train = cfg['batch_size_train']
    bs_test = cfg['batch_size_test']

    # Data
    pretrain_train_loader = get_loader(cfg['pretrained_dataset'], train=True,  batch_size=bs_train, device=device)
    pretrain_test_loader  = get_loader(cfg['pretrained_dataset'], train=False, batch_size=bs_test,  device=device)
    finetune_train_loader = get_loader(cfg['finetuned_dataset'],  train=True,  batch_size=bs_train, device=device)
    finetune_test_loader  = get_loader(cfg['finetuned_dataset'],  train=False, batch_size=bs_test,  device=device)

    # Train target model on pretrained dataset
    print(f"\n--- Training {tgt['name']} on {cfg['pretrained_dataset']} ({tgt['hidden_size_1']} -> {tgt['hidden_size_2']}) ---")
    target_model = build_dense_model(tgt['hidden_size_1'], tgt['hidden_size_2'], use_relu)
    target_model = train_model(target_model, pretrain_train_loader, pretrain_test_loader,
                               num_epochs=cfg['pretrain_epochs'], lr=cfg['pretrain_lr'],
                               device=device, name=f"{tgt['name']}-{cfg['pretrained_dataset']}")

    # Train source model on pretrained dataset
    print(f"\n--- Training {src['name']} on {cfg['pretrained_dataset']} ({src['hidden_size_1']} -> {src['hidden_size_2']}) ---")
    source_model = build_dense_model(src['hidden_size_1'], src['hidden_size_2'], use_relu)
    source_model = train_model(source_model, pretrain_train_loader, pretrain_test_loader,
                               num_epochs=cfg['pretrain_epochs'], lr=cfg['pretrain_lr'],
                               device=device, name=f"{src['name']}-{cfg['pretrained_dataset']}")

    # Finetune target model
    print(f"\n--- Fine-Tuning {tgt['name']} on {cfg['finetuned_dataset']} ---")
    target_finetuned = build_dense_model(tgt['hidden_size_1'], tgt['hidden_size_2'], use_relu)
    target_finetuned.load_state_dict(target_model.state_dict())
    target_finetuned = train_model(target_finetuned, finetune_train_loader, finetune_test_loader,
                                   num_epochs=cfg['finetune_epochs'], lr=cfg['finetune_lr'],
                                   device=device, name=f"{tgt['name']}-{cfg['finetuned_dataset']}")

    # Finetune source model
    print(f"\n--- Fine-Tuning {src['name']} on {cfg['finetuned_dataset']} ---")
    source_finetuned = build_dense_model(src['hidden_size_1'], src['hidden_size_2'], use_relu)
    source_finetuned.load_state_dict(source_model.state_dict())
    source_finetuned = train_model(source_finetuned, finetune_train_loader, finetune_test_loader,
                                   num_epochs=cfg['finetune_epochs'], lr=cfg['finetune_lr'],
                                   device=device, name=f"{src['name']}-{cfg['finetuned_dataset']}")

    # Save
    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    suffix = '_relu' if use_relu else '_norelu'
    torch.save(target_model.state_dict(),      os.path.join(save_dir, f'{tgt["name"]}_model{suffix}.pth'))
    torch.save(source_model.state_dict(),      os.path.join(save_dir, f'{src["name"]}_model{suffix}.pth'))
    torch.save(target_finetuned.state_dict(),  os.path.join(save_dir, f'{tgt["name"]}_finetuned{suffix}.pth'))
    torch.save(source_finetuned.state_dict(),  os.path.join(save_dir, f'{src["name"]}_finetuned{suffix}.pth'))
    print(f"\nModels saved to '{save_dir}/'")

    # Final accuracy summary
    print("\n========== Final Test Accuracies ==========")
    print(f"  [{cfg['pretrained_dataset']} test set]")
    evaluate(target_model,      pretrain_test_loader, device, f"{tgt['name']} pretrained     ")
    evaluate(source_model,      pretrain_test_loader, device, f"{src['name']} pretrained     ")
    evaluate(target_finetuned,  pretrain_test_loader, device, f"{tgt['name']} finetuned      ")
    evaluate(source_finetuned,  pretrain_test_loader, device, f"{src['name']} finetuned      ")

    print(f"  [{cfg['finetuned_dataset']} test set]")
    evaluate(target_model,      finetune_test_loader, device, f"{tgt['name']} pretrained     ")
    evaluate(source_model,      finetune_test_loader, device, f"{src['name']} pretrained     ")
    evaluate(target_finetuned,  finetune_test_loader, device, f"{tgt['name']} finetuned      ")
    evaluate(source_finetuned,  finetune_test_loader, device, f"{src['name']} finetuned      ")


if __name__ == "__main__":
    main()
