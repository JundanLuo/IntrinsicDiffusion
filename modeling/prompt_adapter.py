# ------------------------------------------------------------------------
# This file is part of IntrinsicDiffusion.
# Licensed under the Apache License, Version 2.0.
# See https://github.com/JundanLuo/IntrinsicDiffusion for details.
# If you use this code, please cite the paper listed on the above website.
# ------------------------------------------------------------------------


import os.path

import torch


class PromptAdapter(torch.nn.Module):
    """
    A module that manages text prompt embeddings.
    Supports initialization from either a text encoder or random initialization.
    """
    def __init__(self, tokenizer, text_encoder, prompt_texts):
        """
        Initializes the PromptAdapter with either pretrained embeddings or random tensors.
        """
        super().__init__()
        self.prompt_embeds = torch.nn.ParameterDict()
        for text in prompt_texts:
            if tokenizer is not None or text_encoder is not None:
                assert tokenizer is not None and text_encoder is not None, \
                    "tokenizer and text_encoder must be both None or both not None"
                text_inputs = tokenizer(
                    text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                    return_tensors="pt"
                ).input_ids
                embeds = text_encoder(text_inputs)[0]
                self.prompt_embeds[text] = torch.nn.parameter.Parameter(embeds)
                print(f"Initialized prompt adapter for {text} using text embedding of {text}")
            else:
                self.prompt_embeds[text] = torch.nn.parameter.Parameter(torch.randn(1, 77, 1024))

    def forward(self, text_prompts):
        """
        Forward pass for retrieving the embeddings corresponding to input prompt(s).
        """
        if isinstance(text_prompts, str):
            return self.prompt_embeds[text_prompts]
        assert isinstance(text_prompts, list)
        embeds = [self.prompt_embeds[text] for text in text_prompts]
        return torch.cat(embeds, dim=0)

    def save_pretrained(self, save_directory: str):
        """
        Saves the prompt embeddings to disk.
        """
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "prompt_adapter.pt"))

    def load_pretrained(self, load_directory: str, strict=True):
        """
        Loads the prompt adapter parameters from disk.
        """
        path = os.path.join(load_directory, "prompt_adapter.pt")
        if not os.path.exists(path):
            print(f"WARNING: {path} does not exist. Skipping loading prompt adapter.")
            return
        self.load_state_dict(torch.load(path), strict=strict)
