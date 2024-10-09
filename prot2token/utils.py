import os
import yaml
import torch
from box import Box
from huggingface_hub import hf_hub_url
import requests
from tqdm import tqdm


def load_configs(config, inference=False):
    """
        Load the configuration file and convert the necessary values to floats.

        Args:
            config (dict): The configuration dictionary.
            inference (bool): A boolean flag to indicate if the configuration is for inference.

        Returns:
            The updated configuration dictionary with float values.
        """

    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)

    if not inference:
        # Convert the necessary values to floats.
        tree_config.optimizer.lr = float(tree_config.optimizer.lr)
        tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
        tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
        tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    return tree_config


def download_file_with_progress(repo_id, filename, desc, subfolder):
    """
    Helper function to download a file from Huggingface with a progress bar.

    Args:
        repo_id (str): The Huggingface repository ID.
        filename (str): The file path in the repository.
        desc (str): Description for the progress bar.
        subfolder (str): Subfolder to save the file in.

    Returns:
        local_path (str): The local file path where the file is saved.
    """
    file_url = hf_hub_url(repo_id=repo_id, filename=filename)
    # Create path for the subfolder
    model_dir = os.path.join("ckpt", subfolder)
    os.makedirs(model_dir, exist_ok=True)
    # Save the file in the subfolder
    local_path = os.path.join(model_dir, os.path.basename(filename))

    # Check if file already exists
    if not os.path.exists(local_path):
        # Perform the download with progress tracking
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            chunk_size = 1024  # 1 KB chunks

            with open(local_path, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=desc
            ) as bar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"File downloaded to {local_path}")
    else:
        print(f"File already exists at {local_path}")
    return local_path

def prepare_config_and_checkpoint(name):
    """
    Prepare the configuration dictionary, checkpoint file path, and decoder tokenizer for a model.

    Args:
        name (str): The name of the model ('stability' or 'fluorescence').

    Returns:
        configs: The configuration loaded as a dictionary.
        checkpoint_path: The file path to the checkpoint file.
        decoder_tokenizer_dict: The dictionary containing the decoder tokenizer.
    """
    repo_id = "Mahdip72/prot2token"
    file_paths = {
        'stability': {
            'decoder_tokenizer': "stability/2024-07-05__17-35-31/decoder_tokenizer.yaml",
            'checkpoint': "stability/2024-07-05__17-35-31/checkpoints/best_valid_stability_spearman.pth",
            'config': "stability/2024-07-05__17-35-31/config.yaml"
        },
        'fluorescence': {
            'decoder_tokenizer': "fluorescence/2024-04-23__18-20-05/decoder_tokenizer.yaml",
            'checkpoint': "fluorescence/2024-04-23__18-20-05/checkpoints/best_valid_fluorescence_spearman.pth",
            'config': "fluorescence/2024-04-23__18-20-05/config.yaml"
        }
    }

    if name not in file_paths:
        raise ValueError(f"Model with name '{name}' is not supported.")

    decoder_tokenizer_path = download_file_with_progress(
        repo_id, file_paths[name]['decoder_tokenizer'],
        f"Downloading {name} decoder tokenizer", name
    )
    checkpoint_path = download_file_with_progress(
        repo_id, file_paths[name]['checkpoint'],
        f"Downloading {name} checkpoint", name
    )
    config_file_path = download_file_with_progress(
        repo_id, file_paths[name]['config'],
        f"Downloading {name} config", name
    )

    # Load the configuration file
    with open(config_file_path, 'r') as file:
        dict_config = yaml.full_load(file)

    configs = load_configs(dict_config)

    # Load the decoder tokenizer dictionary
    with open(decoder_tokenizer_path, 'r') as file:
        decoder_tokenizer_dict = yaml.full_load(file)
    return configs, checkpoint_path, decoder_tokenizer_dict


def remove_prefix_from_keys(dictionary, prefix):
    new_dict = {}
    for key, value in dictionary.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    return new_dict


def add_prefix_to_keys(dictionary, prefix):
    new_dict = {}
    for key, value in dictionary.items():
        new_key = prefix + key  # Concatenate the prefix and the original key
        new_dict[new_key] = value  # Store the value with the new key in the new dictionary
    return new_dict


def load_checkpoints_inference(checkpoint_path, net):
    """
    Load a PyTorch checkpoint from a specified path into the provided model.

    Args:
        checkpoint_path (str): The file path to the checkpoint.
        logging (Logger): Logger for logging messages.
        net (torch.nn.Module): The model into which the checkpoint will be loaded.

    Returns:
        torch.nn.Module: The model with loaded checkpoint weights.
    """
    # Check if the checkpoint file exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file does not exist at {checkpoint_path}")
        return net

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Removing the prefix "_orig_mod." from the keys of the model checkpoint if it exists.
    checkpoint['model_state_dict'] = remove_prefix_from_keys(checkpoint['model_state_dict'],
                                                             '_orig_mod.')

    # Load state dict into the model
    load_log = net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f'Loading checkpoint log: {load_log}')
    return net


class InferenceTokenizer:
    def __init__(self, token_dict):
        self.tokens_dict = token_dict
        self.index_token_dict = self.update_index_token_dict()
        self.vocab_size = len(self.tokens_dict)

    def update_index_token_dict(self):
        return {value: key for key, value in self.tokens_dict.items()}

    def __call__(self,
                 task_name: int):
        encoded_target = [self.tokens_dict['<bos>'], self.tokens_dict[task_name]]
        return encoded_target


if __name__ == '__main__':
    # For test utils modules
    print('done')
