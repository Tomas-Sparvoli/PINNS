import torch


def main() -> None:
    print("Torch:", torch.__version__)
    print("CUDA disponible:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
