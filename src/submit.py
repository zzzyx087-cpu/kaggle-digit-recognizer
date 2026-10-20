import pandas as pd
import torch

from model import CNN_BN

model = CNN_BN()

model.load_state_dict(
    torch.load(
        "outputs/best_cnn_combined.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

test_df = pd.read_csv("data/test.csv")

X_test = torch.tensor(
    test_df.to_numpy(),
    dtype=torch.float32
) / 255.0

X_test = X_test.reshape(-1, 1, 28, 28)

predictions = []

with torch.no_grad():
    for images in X_test.split(256):
        outputs = model(images)
        predictions.extend(outputs.argmax(dim=1).tolist())

submission = pd.DataFrame({
    "ImageId": range(1, len(predictions) + 1),
    "Label": predictions
})

submission.to_csv("submission.csv", index=False)

print(submission.head())
print(f"Total predictions: {len(predictions)}")