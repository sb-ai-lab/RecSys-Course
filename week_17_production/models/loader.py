import importlib.util
from pathlib import Path
from typing import Optional

_MODEL_MODULE_NAME = "model"
_MODEL_CLASS_NAME = "Model"


def model_class_for_inference(
    model_name: str, module_name: Optional[str] = None, model_class_name: Optional[str] = None
):
    module_name = module_name or _MODEL_MODULE_NAME
    model_class_name = model_class_name or _MODEL_CLASS_NAME

    base_path = Path(__file__).absolute().parent
    model_module_path = base_path / model_name / (module_name + ".py")

    # We change actual class module name to let model to be correctly unpickled
    # at the loading phase. The problem is that the code paths are added to the
    # MLFlow archive without parent directory, so MLFLow cannot correctly
    # reconstruct fully qualified names of the class
    model_module_spec = importlib.util.spec_from_file_location(f"{model_name}.{module_name}", model_module_path)

    model_module = importlib.util.module_from_spec(model_module_spec)
    model_module_spec.loader.exec_module(model_module)

    model = getattr(model_module, model_class_name)

    # The added code paths are expected to be used when
    # `mlflow.{flavour}.save_mode` is called
    model.code_path = [str(model_module_path.parent)]
    model.config_path = model_module_path.parent / "config.yml"

    return model
