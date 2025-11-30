import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from logger import Logger

log = Logger("Model Loader Logs", True, "Logs/model.log", "DEV")

class ModelHandler:
    def __init__(self, model_name: str, quantize: bool = False):
        """
        Initialization of class arguments.

        1. model_name -> str -> Hugging face repo id.\n
        2. quantize -> bool -> Whether to not quantize the model.\n
        3. device -> str -> Select the device on which you want to load the model, e.g., "cpu" or "cuda" or "mps"\n
        """
        self.model_name = model_name
        self.quantize = quantize
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        log.debug("Initialisation done successfully!")
    
    def load_model(self):
        """
        Loads the tokenizer and model according to the initialization parameters.

        Returns:
            model: The loaded AutoModelForCausalLM on the specified device.
            tokenizer: The corresponding AutoTokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        log.debug("Model's Tokenizer successfull loaded!")
        if self.quantize:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                dtype=torch.float16,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
        
        log.debug("Model Loaded successfully!")
        return model, tokenizer
