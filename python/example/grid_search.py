import mlflow.pytorch
from sklearn.model_selection import ParameterGrid

from train_mlflow import parse_args, init_experiment, train_evaluation


def grid_search(args, hparam_grid):

    hparam_groups = ParameterGrid(hparam_grid)

    best_hparams = None
    best_metric = 0

    with mlflow.start_run(run_name="Parent"):

        for key, hparams in hparam_grid.items():
            mlflow.set_tag(key, hparams)    # log grid

        for i, hparams in enumerate(hparam_groups):
            with mlflow.start_run(nested=True, run_name=f"Trial {i}") as child_run:

                test_acc = train_evaluation(child_run, args, hparams)

                # select the best hparams
                if test_acc > best_metric:
                    best_metric = test_acc
                    best_hparams = hparams

        mlflow.log_metric("best_metric", best_metric)
        for key, value in best_hparams.items():
            mlflow.log_param(key, value)


def main():
    args = parse_args()
    init_experiment(args)

    hparam_grid = {
        'feat_out1': [8, 16],
        'feat_out2': [16],
        'feat_out3': [8, 16],
        'clf_hid': [32, 64],
        'feat_lr': [1e-2],
        'clf_lr': [1e-2, 1e-3],
        'batch_size': [32, 64]
    }   # 32 trails
    grid_search(args, hparam_grid)


if __name__ == "__main__":
    main()
