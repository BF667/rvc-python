import os
import logging
from glob import glob
import soundfile as sf
import torch
from rvc_python.modules.vc.modules import VC
from rvc_python.configs.config import Config
from rvc_python.download_model import download_rvc_models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RVCInference:
    def __init__(self, models_dir="rvc_models", model_path=None, index_path="", version="v2"):
        self.models_dir = models_dir
        self.lib_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = self._detect_device()
        self.config = Config(self.lib_dir, self.device)
        self.vc = VC(self.lib_dir, self.config)
        self.current_model = None
        self.models = {}

        # Default parameters
        self.params = {
            'f0method': 'harvest',
            'f0up_key': 0,
            'index_rate': 0.5,
            'filter_radius': 3,
            'resample_sr': 0,
            'rms_mix_rate': 1,
            'protect': 0.33
        }

        # Download models if necessary
        download_rvc_models(self.lib_dir)

        # Load available models
        self.models = self._load_available_models()

        # Load model if provided
        if model_path:
            self.load_model(model_path, version, index_path)

    def _detect_device(self):
        """Detects and returns the best available device (GPU or CPU)."""
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected")
            return "cuda:0"
        logger.info("Using CPU")
        return "cpu:0"

    def _load_available_models(self):
        """Loads available models from directory with optimized path handling."""
        models = {}
        for model_dir in glob(os.path.join(self.models_dir, "*")):
            if not os.path.isdir(model_dir):
                continue
            model_name = os.path.basename(model_dir)
            pth_file = next(iter(glob(os.path.join(model_dir, "*.pth"))), None)
            index_file = next(iter(glob(os.path.join(model_dir, "*.index"))), None)
            if pth_file:
                models[model_name] = {"pth": pth_file, "index": index_file}
        return models

    def set_models_dir(self, new_models_dir):
        """Sets new models directory with validation."""
        if not os.path.isdir(new_models_dir):
            logger.error(f"Directory {new_models_dir} does not exist")
            raise ValueError(f"Directory {new_models_dir} does not exist")
        self.models_dir = new_models_dir
        self.models = self._load_available_models()
        logger.info(f"Models directory set to {new_models_dir}")

    def list_models(self):
        """Returns list of available model names."""
        return list(self.models.keys())

    def load_model(self, model_path_or_name, version="v2", index_path=""):
        """Loads a model with optimized resource management."""
        try:
            model_info = self.models.get(model_path_or_name, {})
            if model_path_or_name in self.models:
                model_path = model_info["pth"]
                index_path = model_info.get("index", "")
                model_name = model_path_or_name
            else:
                model_path = model_path_or_name
                model_name = os.path.basename(model_path)
                if index_path and not os.path.isfile(index_path):
                    logger.error(f"Index file {index_path} not found")
                    raise ValueError(f"Index file {index_path} not found")
                self.models[model_name] = {"pth": model_path, "index": index_path}

            if not os.path.isfile(model_path):
                logger.error(f"Model file {model_path} not found")
                raise ValueError(f"Model file {model_path} not found")

            self.vc.get_vc(model_path, version)
            self.current_model = model_name
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def unload_model(self):
        """Unloads the current model with cleanup."""
        if not self.current_model:
            logger.warning("No model loaded")
            return
        try:
            self.vc = VC(self.lib_dir, self.config)
            self.current_model = None
            logger.info("Model unloaded successfully")
        except Exception as e:
            logger.error(f"Failed to unload model: {str(e)}")
            raise

    def set_params(self, **kwargs):
        """Sets inference parameters with validation."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                logger.warning(f"Ignoring unrecognized parameter: {key}")

    def infer_file(self, input_path, output_path):
        """Processes a single audio file with context management."""
        if not self.current_model:
            logger.error("No model loaded")
            raise ValueError("No model loaded")

        try:
            model_info = self.models[self.current_model]
            file_index = model_info.get("index", "")

            with torch.no_grad():
                wav_opt = self.vc.vc_single(
                    sid=0,
                    input_audio_path=input_path,
                    f0_up_key=self.params['f0up_key'],
                    f0_method=self.params['f0method'],
                    file_index=file_index,
                    index_rate=self.params['index_rate'],
                    filter_radius=self.params['filter_radius'],
                    resample_sr=self.params['resample_sr'],
                    rms_mix_rate=self.params['rms_mix_rate'],
                    protect=self.params['protect'],
                    f0_file="",
                    file_index2=""
                )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, wav_opt, self.vc.tgt_sr)
            logger.info(f"Processed file saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to process file {input_path}: {str(e)}")
            raise

    def infer_dir(self, input_dir, output_dir):
        """Processes all audio files in a directory."""
        if not self.current_model:
            logger.error("No model loaded")
            raise ValueError("No model loaded")

        try:
            os.makedirs(output_dir, exist_ok=True)
            audio_files = [f for f in glob(os.path.join(input_dir, "*.*")) if os.path.isfile(f)]
            processed_files = []

            for input_path in audio_files:
                output_filename = os.path.splitext(os.path.basename(input_path))[0] + '.wav'
                output_path = os.path.join(output_dir, output_filename)
                processed_files.append(self.infer_file(input_path, output_path))

            logger.info(f"Processed {len(processed_files)} files")
            return processed_files
        except Exception as e:
            logger.error(f"Failed to process directory {input_dir}: {str(e)}")
            raise

    def set_device(self, device):
        """Sets the computation device."""
        try:
            self.device = device
            self.config.device = device
            self.vc.device = device
            logger.info(f"Device set to {device}")
        except Exception as e:
            logger.error(f"Failed to set device: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        rvc = RVCInference(
            model_path="path/to/model.pth",
            index_path="path/to/index.index",
            version="v2"
        )
        rvc.set_params(f0up_key=2, protect=0.5)
        rvc.infer_file("input.wav", "output.wav")
        rvc.infer_dir("input_dir", "output_dir")
        rvc.unload_model()
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
