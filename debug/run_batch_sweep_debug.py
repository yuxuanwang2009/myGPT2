import config
import run_train_debug


def main() -> None:
    batch_sizes = [32, 64, 8, 16]
    for batch_size in batch_sizes:
        config.cfg.dropout = 0.0
        config.cfg.grad_clipping = 0.0
        config.cfg.batch_size = batch_size
        config.batch_size = batch_size
        run_train_debug.main()


if __name__ == "__main__":
    main()
