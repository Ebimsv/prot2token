import torch
import pandas as pd
from models import prepare_models


def test():
    # Clear CUDA cache
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")

    # Prepare models
    net, decoder_tokenizer, inference_configs = prepare_models(name="stability", device="cpu", compile_model=False)
    print("Models prepared:")
    print(f"Net: {net}")
    print(f"Decoder Tokenizer: {decoder_tokenizer}")
    print(f"Inference Configs: {inference_configs}")

    # Load samples from CSV
    samples = pd.read_csv("../data/inference_data/stability_inference.csv")["input"].tolist()
    print("Samples loaded:")
    print(samples)

    # Run inference
    results = net.run(samples, merging_character="")
    print("Inference results:")
    print(results)

    # Clear CUDA cache again
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")

if __name__ == "__main__":
    test()
    print("done!")
