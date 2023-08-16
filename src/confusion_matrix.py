# import model.py
import model
from model import PatchedFlairModule
from torch.utils.data import DataLoader


def create_confusion_matrix():
    checkpoint = 'tb_logs/test1/version_1/checkpoints/epoch=12-step=10075.ckpt'
    model = PatchedFlairModule.load_from_checkpoint(checkpoint_path=checkpoint)

    # set model to eval mode
    model.eval()

    # creat pred array
    predictions = []

    for batch in test_dataloader:
        inputs = batch  # Passe dies entsprechend deiner Daten an
        with torch.no_grad():
            outputs = model(inputs)
            _, batch_predictions = torch.max(outputs, 1)  # Annahme: Klassen sind entlang der zweiten Dimension
            predictions.extend(batch_predictions.cpu().numpy())


# define main
def main():
    # create confusion matrix
    create_confusion_matrix()


# run main
if __name__ == '__main__':
    main()
