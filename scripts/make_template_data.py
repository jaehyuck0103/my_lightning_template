from pathlib import Path

from torchvision import datasets

DATA_ROOT = Path(__file__).parent.parent / "Data"

TRAIN_IMG_DIR = DATA_ROOT / "train_images"
TRAIN_CSV_PATH = DATA_ROOT / "train.csv"
TEST_IMG_DIR = DATA_ROOT / "test_images"


def main():
    train_dataset = datasets.MNIST(DATA_ROOT, train=True, download=True)
    test_dataset = datasets.MNIST(DATA_ROOT, train=False, download=True)

    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_CSV_PATH, "w") as fp:
        for idx, item in enumerate(train_dataset):
            basename = f"{idx:06d}"
            img_path = TRAIN_IMG_DIR / f"{basename}.jpg"

            # Save image and label
            item[0].save(img_path)
            fp.write(f"{basename}, {item[1]}\n")

    for idx, item in enumerate(test_dataset):
        basename = f"{idx:06d}"
        img_path = TEST_IMG_DIR / f"{basename}.jpg"

        # Save image
        item[0].save(img_path)


if __name__ == "__main__":
    main()
