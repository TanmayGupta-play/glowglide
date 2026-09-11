from pathlib import Path
import shutil
import kagglehub


DATASET = "nadyinky/sephora-products-and-skincare-reviews"


def main():
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Sephora dataset...")

    downloaded_path = Path(
        kagglehub.dataset_download(DATASET)
    )

    print("Downloaded to:", downloaded_path)

    csv_files = list(downloaded_path.glob("*.csv"))

    print("\nCSV files found:")

    for file in csv_files:
        print("-", file.name)

        destination = raw_dir / file.name

        shutil.copy2(file, destination)

    print("\nDataset copied successfully to:")
    print(raw_dir)


if __name__ == "__main__":
    main()