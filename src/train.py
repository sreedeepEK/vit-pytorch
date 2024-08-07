import os
import torch
import src.data_setup as data_setup
import src.engine as engine
import src.utils as utils
from src.logger import global_logger as logger
from torchvision import transforms
import src.model as model_module

def main():

    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    train_dir = "data\\retinal_oct\\train"
    test_dir = "data\\retinal_oct\\test"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use the transformations required by ResNet50
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )
    logger.info("Data transformed successfully.")

    # Initialize the ResNet50 model
    model, _ = model_module.resnet_model(num_classes=len(class_names))
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device
    )
    
    utils.save_model(
        model=model,
        target_dir="models",
        model_name="model.pth"
    )
    logger.info("Model trained successfully.")
    logger.info("Model saved to models folder.")

if __name__ == '__main__':
    main()
