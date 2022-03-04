import mlflow.pytorch
from sklearn.model_selection import ParameterGrid

from train_mlflow import make_reproducible, parse_args, train_evaluation


def grid_search(args, hparam_grid):

    hparam_groups = ParameterGrid(hparam_grid)

    best_hparams = None
    best_metric = 0

    with mlflow.start_run(run_name="Parent") as parent_run:
        parent_artifact_path = parent_run.info.artifact_uri

        for key, hparams in hparam_grid.items():
            mlflow.set_tag(key, hparams)    # log grid

        for i, hparams in enumerate(hparam_groups):
            with mlflow.start_run(nested=True, run_name=f"Trial {i}") as child_run:

                child_artifact_path = child_run.info.artifact_uri

                hp_metric = train_evaluation(
                    child_run, args, hparams,
                    log_dirpath=parent_artifact_path,
                    ckpt_dirpath=child_artifact_path)

                # select the best hparams
                if hp_metric > best_metric:
                    best_metric = hp_metric
                    best_hparams = hparams

        mlflow.log_metric("best_metric", best_metric)
        for key, value in best_hparams.items():
            mlflow.log_param(key, value)


def main():
    args = parse_args()
    make_reproducible(args)

    hparam_grid = {
        'feat_out1': [8, 16],
        'feat_out2': [16],  # [16, 32],
        'feat_out3': [8, 16],
        'clf_hid': [32],  # [32, 64],
        'feat_lr': [1e-2],  # [1e-2, 5e-3],
        'clf_lr': [1e-3],  # [1e-2, 1e-3],
        'batch_size': [32, 64]
    }
    grid_search(args, hparam_grid)


if __name__ == "__main__":
    main()
