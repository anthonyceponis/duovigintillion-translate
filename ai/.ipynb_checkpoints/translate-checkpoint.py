import torch
import torch.nn as nn
from torch.utils.data import Dataset

from model import build_transformer

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

class TranslationDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq):
        super().__init__()
        self.seq = seq # the sequence length to which all sequences will be truncated/padded to.

        self.ds = ds # The dataset containing the source and target pairs.
        self.tokenizer_src = tokenizer_src # Tokenizer for the source language.
        self.tokenizer_tgt = tokenizer_tgt # Tokenizer for the target language.
        self.src_lang = src_lang # Source language identifier e.g. "en" for english.
        self.tgt_leng = tgt_lang # Target language identifier e.g. "it" for italian.

        # Generate vocab ids for our special tokens using the target tokenizer.
        # We use the target tokenizer and not the source tokenizer because these tags are used by the decoder and not the encoder.
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Get the raw text.
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_leng]

        # Transform the raw text into tokens.
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculate how much padding to add to tokenized texts.
        enc_num_padding_tokens = self.seq - len(enc_input_tokens) - 2 # We will add [SOS] and [EOS] to the tokenized input so subtract 2 to calc padding.
        dec_num_padding_tokens = self.seq - len(dec_input_tokens) - 1 # We only add [SOS] for the decoder input (remember shifted right). In contrast, the label for the decoder output will only have [EOS]. Hence, we subtract just 1.

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long. Note that truncation generally does not work well, so better to just give up.
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add [SOS] and [EOS] tokens to encoder.
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Add only [SOS] token to decoder.
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Add only [EOS] token to label.
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Double check the size of the tensors to make sure they are all seq long
        assert encoder_input.size(0) == self.seq
        assert decoder_input.size(0) == self.seq
        assert label.size(0) == self.seq

        # 1. We add dimensionality in comments for each line for clarity.
        # 2. Encoder mask is just a boolean vector to zero out padding tokens in the self attention matrix later. 
        #    Note that it gets stretched from (seq) to (1, 1, seq) to add dimensionality.
        # 3. Decoder mask does same as encoder mask, but only gets stretched once. 
        #    We also apply a casual mask, which prevents tokens from seeing into the future. 
        #    Note that the padding mask is broadcasted (1, seq) -> (1, seq, seq) where the seq is duplicated so it could then be bitwise ANDed with the casual mask.
        return {
            "encoder_input": encoder_input, # (seq)
            "decoder_input": decoder_input, # (seq)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1, seq) & (1, seq, seq) -> (1, seq, seq)
            "label": label, # (seq)
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def casual_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0 # Note that we created an upper triangular matrix, whereas we need a lower one, so we flip all bits.

from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq": 350,
        "d_model": 512,
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

# Define generator to get sentences from a lang.
def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # Create a word level based tokenizer model, specifying [UNK] as the token for unkown words.
        tokenizer.pre_tokenizer = Whitespace() # Set pre tokenizer to split on whitespace.
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) # Create trainer for our word level tokenizer, specifying our special tokens, with a min freq threshold of 2. This means that a word will only be included if it appears at least twice across the training corpus.
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer) # Specifies how to get data for training, which in our case is using out iterator.
        tokenizer.save(str(tokenizer_path)) # Remember that the tokenizer is a trained model, so we can save it to load it later.
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path)) # Load a pre trained tokenizer.
    return tokenizer

import torch.nn.functional as F

def beam_search_decode(model, tokenizer_src, tokenizer_tgt, input_sequence, max_len, device, beam_width=4):
    # Encode the input sequence
    encoder_input = torch.tensor(tokenizer_src.encode(input_sequence).ids).unsqueeze(0).to(device)
    encoder_mask = (encoder_input != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(2).to(device)
    encoder_output = model.encode(encoder_input, encoder_mask)  # (1, seq_len, d_model)

    # Initialize the beams
    beams = [(torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], device=device), 0)]  # (sequence tensor, score)
    completed_sequences = []

    for _ in range(max_len):
        # Prepare candidates
        all_candidates = []
        for seq, score in beams:
            if seq[-1] == tokenizer_tgt.token_to_id("[EOS]"):  # If EOS is reached, save sequence
                completed_sequences.append((seq, score))
                continue

            # Create decoder input and mask
            decoder_input = seq.unsqueeze(0)
            decoder_mask = torch.tril(torch.ones((1, len(seq), len(seq)), device=device)).bool()

            # Decode and project
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output[:, -1, :])  # Only the last token

            # Get the top-k tokens and their scores
            log_probs = F.log_softmax(proj_output, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

            # Expand each current sequence in the beam with the top-k tokens
            for k in range(beam_width):
                next_seq = torch.cat([seq, topk_indices[0, k].unsqueeze(0)])
                next_score = score + topk_log_probs[0, k].item()
                all_candidates.append((next_seq, next_score))

        # Sort all candidates by score and select the best `beam_width` ones
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        beams = ordered[:beam_width]

        # Stop if all beams have finished (contain [EOS])
        if all(seq[-1] == tokenizer_tgt.token_to_id("[EOS]") for seq, _ in beams):
            break

    # Add remaining beams to completed sequences and select the one with the highest score
    completed_sequences.extend(beams)
    best_sequence = sorted(completed_sequences, key=lambda tup: tup[1], reverse=True)[0][0]

    # Decode the best sequence back into text
    decoded_tokens = [tokenizer_tgt.id_to_token(id.item()) for id in best_sequence]
    return decoded_tokens

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq"], config['seq'], d_model=config['d_model'])
    return model

def translate_sentence(sentence, verbose=False):
    # Define the device.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print("Using device:", device)
        
        if device == "cuda":
            print(f"Device name: {torch.cuda.get_device_name(device.index)}")
            print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
        else:
            print("NOTE: If you have a GPU, consider using it for trainig.")

    device = torch.device(device)
    
    config = get_config()
    dataset_raw = [
        {"translation": {
            "en": sentence, 
            "it": ""
        }}
    ]
    # Build tokenizers.
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config["lang_tgt"])

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    preload = config["preload"]
    model_filename = latest_weights_file_path(config) if preload == "latest" else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        if verbose:
            print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    else:
        raise Exception("No model to preload, please specify model to load.")
    
    answer_ds = TranslationDataset(dataset_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq"])
    answer_dataloader = DataLoader(answer_ds, batch_size=1, shuffle=True)
    
    with torch.no_grad():
        # Set up a progress bar
        for batch in answer_dataloader:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            # Ensure batch size is 1 for validation
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            max_len=config["seq"]
            beam_width=4
            
            # Run beam search decoding
            model_out = beam_search_decode(
                model, tokenizer_src, tokenizer_tgt, batch["src_text"][0], max_len, device, beam_width=beam_width
            )
            
            # Remove special tokens from the decoded text
            model_out_text = ' '.join([token for token in model_out if token not in {"[SOS]", "[EOS]", "[PAD]"}])
            
            return model_out_text