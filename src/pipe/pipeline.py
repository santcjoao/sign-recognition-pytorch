import dataclasses
import json
import os

import torch
import torchvision.transforms as T

from src.core.types import ExperimentPlan, ExperimentResult, ExperimentSuite
from src.models import build_model
from src.pipe.testing import Tester
from src.pipe.training import Trainer
from src.utils.gtsrb import _MEAN, _STD, get_dataloaders, save_predictions

BASE_TRANSFORM = T.Compose(
    [
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ]
)


def get_transform(transform: T.Compose | None = None) -> T.Compose:
    if transform is None:
        return BASE_TRANSFORM
    return T.Compose(
        [
            T.Resize((32, 32)),
            transform,
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ]
    )


class PipelineError(Exception):
    pass


class Pipeline:
    def __init__(self, suite: ExperimentSuite, transform: T.Compose | None = None):
        self.suite = suite
        self.transform = transform
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self) -> None:
        print(f"Hardware: {self.device.upper()}")
        print(f"Running suite with {len(self.suite.experiment_plans)} plans...")

        os.makedirs("results", exist_ok=True)

        for plan in self.suite:
            self._execute_plan(plan)

    def _execute_plan(self, plan: ExperimentPlan) -> None:
        print(f"\n{'=' * 50}\nStarting experiment: {plan.experiment_name}\n{'=' * 50}")

        # Data Augmentation
        if plan.model_config.data_augmentation:
            if self.transform is None:
                raise PipelineError(
                    "Transform must be provided when data_augmentation is True"
                )
            current_transform = get_transform(self.transform)
        else:
            current_transform = get_transform()

        # Get Dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            img_size=32,
            batch_size=plan.hyperparameters.batch_size,
            train_transform=current_transform,
        )

        # Model Creation
        model = build_model(plan.model_config).to(self.device)

        # Stages
        ## Training
        trainer = Trainer(model, plan.hyperparameters, self.device)
        history = trainer.train(train_loader, val_loader)

        ## Testing
        tester = Tester(model, self.device)
        test_preds = tester.test(test_loader)

        # Serialize Results
        result = ExperimentResult(
            plan=plan,
            history=history,
            test_predictions=test_preds,
            model_state_path=os.path.join(plan.run_path, "best_model.pth"),
        )
        self._serialize(result, model)

    def _serialize(self, result: ExperimentResult, model: torch.nn.Module) -> None:
        os.makedirs(result.plan.run_path, exist_ok=True)

        # Save Config
        with open(os.path.join(result.plan.run_path, "config.json"), "w") as f:
            json.dump(dataclasses.asdict(result.plan), f, indent=4)

        # Save History
        with open(os.path.join(result.plan.run_path, "history.json"), "w") as f:
            json.dump(dataclasses.asdict(result.history), f, indent=4)

        # Save Weights
        if result.model_state_path is not None:
            torch.save(model.state_dict(), result.model_state_path)

        # Save Predictions
        import numpy as np

        save_predictions(
            y_pred=np.array(result.test_predictions.y_pred),
            filepath=os.path.join(result.plan.run_path, "predictions.csv"),
            experiment_name=result.plan.experiment_name,
        )
        print(f"Results saved in: {result.plan.run_path}")
