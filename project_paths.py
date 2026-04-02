from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = ROOT_DIR / "workspace"
WORKSPACE_MODELS_DIR = WORKSPACE_DIR / "models"
WORKSPACE_VIDEOS_DIR = WORKSPACE_DIR / "videos"
WORKSPACE_INPUT_VIDEOS_DIR = WORKSPACE_VIDEOS_DIR / "input"
WORKSPACE_OUTPUTS_DIR = WORKSPACE_DIR / "outputs"
WORKSPACE_LOGS_DIR = WORKSPACE_DIR / "logs"
WORKSPACE_CACHE_DIR = WORKSPACE_DIR / "cache"
WORKSPACE_ARCHIVE_SCRIPTS_DIR = WORKSPACE_DIR / "scripts" / "archive"
WORKSPACE_NOTES_DIR = WORKSPACE_DIR / "notes"

ROOT_LABELS_PATH = ROOT_DIR / "labels.txt"
ROOT_DEEPSTREAM_APP_CONFIG = ROOT_DIR / "deepstream_app_config.txt"
ROOT_TRACKER_FALLBACK_CONFIG = ROOT_DIR / "config_tracker_NvDCF_accuracy_no_reid.yml"
ROOT_PGIE_CONFIG_YOLO11 = ROOT_DIR / "config_infer_primary_yolo11.txt"
ROOT_PGIE_CONFIG_YOLO26 = ROOT_DIR / "config_infer_primary_yolo26.txt"
ROOT_SAHI_PREPROCESS_CONFIG = ROOT_DIR / "config_preprocess_sahi.txt"
ROOT_SAHI_INFER_CONFIG = ROOT_DIR / "config_infer_primary_yolo11_sahi.txt"
ROOT_DEEPSTREAM_TRACKER_CONFIG = ROOT_DIR / "deepstream_tracker_config.txt"

DEFAULT_VIDEO_INPUT = WORKSPACE_INPUT_VIDEOS_DIR / "video.mp4"
DEFAULT_TRT_ENGINE = WORKSPACE_MODELS_DIR / "model_b1_gpu0_fp32.engine"
PYTHON_TEST_OUTPUT_DIR = WORKSPACE_OUTPUTS_DIR / "python_test"
TENSORRT_OUTPUT_DIR = WORKSPACE_OUTPUTS_DIR / "tensorrt_test"
ONNX_ENGINE_CACHE_FILE = WORKSPACE_CACHE_DIR / ".onnx_engine_cache.json"

WORKSPACE_DIRS = (
    WORKSPACE_MODELS_DIR,
    WORKSPACE_INPUT_VIDEOS_DIR,
    WORKSPACE_OUTPUTS_DIR,
    WORKSPACE_LOGS_DIR,
    WORKSPACE_CACHE_DIR,
    WORKSPACE_ARCHIVE_SCRIPTS_DIR,
    WORKSPACE_NOTES_DIR,
    PYTHON_TEST_OUTPUT_DIR,
    TENSORRT_OUTPUT_DIR,
)


def ensure_workspace_dirs() -> None:
    for directory in WORKSPACE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


ensure_workspace_dirs()
