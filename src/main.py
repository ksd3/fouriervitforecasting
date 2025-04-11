import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import config
from dataset import WeatherSubset
from model.fourier_vit import FourierViT
from train import train_epoch
from evaluate import evaluate

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = WeatherSubset(config['data_path'], config['max_data_mb'])
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = FourierViT(
        img_size=(256,512),
        patch_size=16,
        atmos_vars=len(config['atmos_vars']),
        atmos_levels=config['atmos_levels'],
        surface_vars=len(config['surface_vars']),
        embed_dim=512,
        fno_modes=(32,32),
        fno_width=98,
        fno_depth=2,
        vit_depth=4,
        vit_heads=8
    ).to(device)

    criterion = nn.MSELoss()
    params = [
        {'params': model.surface_fno.parameters(), 'lr': 1e-4},
        {'params': model.atmos_fno.parameters(), 'lr': 5e-5},
        {'params': model.lead_time_mlp.parameters(), 'lr': 2e-4},
        {'params': model.blocks.parameters()},
        {'params': model.decoder.parameters()}
    ]
    optimizer = optim.AdamW(params, lr=1e-4)

    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f}")

    test_loss = evaluate(model, dataloader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}")

    torch.save(model.state_dict(), "fourier_vit_improved.pth")

if __name__ == "__main__":
    main()
