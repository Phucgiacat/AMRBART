#!/usr/bin/env python3
"""
Two-Stage Inference for AMRBART-RGL

Stage 1: Run normal inference (text → rough AMR) with text as NRL placeholder
Stage 2: Convert rough AMR → NRL tokens, feed back as real NRL input → refined AMR

Usage:
    python two_stage_inference.py \
        --model_name_or_path /path/to/checkpoint \
        --test_file /path/to/test.jsonl \
        --output_dir /path/to/output \
        --tokenizer_name xfbai/AMRBART-large-v2
"""

import os
import sys
import re
import json
import math
import argparse
import logging
import penman
import torch
import numpy as np
from tqdm import tqdm

# Add fine-tune dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_interface.tokenization_bart import AMRBartTokenizer
from model_interface.modeling_bart import BartForConditionalGeneration
from common import postprocessing
from common.penman_interface import encode as penman_encode_custom

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# NRL Conversion Utilities
# ============================================================

def _tokenize_encoded_graph(encoded):
    """Tokenize a penman-encoded string into individual tokens."""
    linearized = re.sub(r'(\".+?\")', r" \1 ", encoded)
    pieces = []
    for piece in linearized.split():
        if piece.startswith('"') and piece.endswith('"'):
            pieces.append(piece)
        else:
            piece = piece.replace("(", " ( ").replace(")", " ) ").replace(":", " :").replace("/", " / ").strip()
            pieces.append(piece)
    return re.sub(r"\s+", " ", " ".join(pieces)).strip().split(" ")


def reverse_tree(tree_node):
    """Recursively reverse the order of children at each node in a penman Tree."""
    if not isinstance(tree_node, tuple) or len(tree_node) != 2:
        return tree_node
    new_branches = []
    for role, target in reversed(tree_node[1]):
        if isinstance(target, tuple) and len(target) == 2 and isinstance(target[1], list):
            new_branches.append((role, reverse_tree(target)))
        else:
            new_branches.append((role, target))
    return (tree_node[0], new_branches)


def dfs_linearize_with_pointers(graph):
    """Linearize a penman graph into tokens with <pointer:N> references (NLR order)."""
    linearized = penman.encode(graph)
    linearized_nodes = _tokenize_encoded_graph(linearized)
    remap = {}
    for i in range(1, len(linearized_nodes)):
        if linearized_nodes[i] == "/":
            remap[linearized_nodes[i - 1]] = f"<pointer:{len(remap)}>"
    i = 1
    linearized_nodes_ = [linearized_nodes[0]] if len(linearized_nodes) > 0 else []
    while i < len(linearized_nodes):
        nxt = linearized_nodes[i]
        if nxt in remap:
            if linearized_nodes_[-1] == "(" and i + 1 < len(linearized_nodes) and linearized_nodes[i + 1] == "/":
                nxt, i = remap[nxt], i + 1
            elif linearized_nodes_[-1].startswith(":"):
                nxt = remap[nxt]
        linearized_nodes_.append(nxt)
        i += 1
    return linearized_nodes_


def dfs_linearize_reverse_with_pointers(graph):
    """Linearize a penman graph into tokens with <pointer:N> references (NRL order = reversed children)."""
    tree = penman.configure(graph)
    reversed_tree_obj = penman.layout.Tree(reverse_tree(tree.node))
    linearized = penman.format(reversed_tree_obj)
    linearized_nodes = _tokenize_encoded_graph(linearized)
    remap = {}
    for i in range(1, len(linearized_nodes)):
        if linearized_nodes[i] == "/":
            remap[linearized_nodes[i - 1]] = f"<pointer:{len(remap)}>"
    i = 1
    linearized_nodes_ = [linearized_nodes[0]] if len(linearized_nodes) > 0 else []
    while i < len(linearized_nodes):
        nxt = linearized_nodes[i]
        if nxt in remap:
            if linearized_nodes_[-1] == "(" and i + 1 < len(linearized_nodes) and linearized_nodes[i + 1] == "/":
                nxt, i = remap[nxt], i + 1
            elif linearized_nodes_[-1].startswith(":"):
                nxt = remap[nxt]
        linearized_nodes_.append(nxt)
        i += 1
    return linearized_nodes_


def penman_to_nrl_tokens(penman_string):
    """Convert a PENMAN-formatted AMR string to NRL token list.
    
    Returns:
        list of str: NRL tokens like ['(', '<pointer:0>', 'concept', ':edge', ...]
        Returns None if parsing fails.
    """
    try:
        graph = penman.decode(penman_string)
        tokens = dfs_linearize_reverse_with_pointers(graph)
        return tokens
    except Exception as e:
        logger.warning(f"Failed to convert AMR to NRL: {e}")
        return None


# ============================================================
# Inference Logic
# ============================================================

def load_test_sentences(test_file):
    """Load test sentences from a JSONL file."""
    sentences = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            sentences.append(data["sent"])
    return sentences


def prepare_text_input(tokenizer, sentences, max_length=400, unified_input=True):
    """Prepare text inputs with unified input format."""
    all_input_ids = []
    for sent in sentences:
        raw_ids = tokenizer(
            sent, max_length=max_length, padding=False, truncation=True
        )["input_ids"]
        if unified_input:
            txt_ids = raw_ids[:max_length - 3] + [
                tokenizer.amr_bos_token_id,
                tokenizer.mask_token_id,
                tokenizer.amr_eos_token_id,
            ]
        else:
            txt_ids = raw_ids
        all_input_ids.append(txt_ids)
    return all_input_ids


def prepare_nrl_input(tokenizer, nrl_token_lists, max_length=400, unified_input=True):
    """Prepare NRL AMR token inputs with unified input format.
    
    Args:
        nrl_token_lists: list of list of str, each inner list is NRL tokens for one sentence
    """
    all_nrl_ids = []
    for nrl_tokens in nrl_token_lists:
        nrl_ids = tokenizer.tokenize_amr(nrl_tokens)[:max_length - 5]
        nrl_ids.append(tokenizer.amr_eos_token_id)
        if unified_input:
            nrl_ids = [
                tokenizer.bos_token_id,
                tokenizer.mask_token_id,
                tokenizer.eos_token_id,
                tokenizer.amr_bos_token_id,
                tokenizer.amr_dfs_NRL_token_id,
            ] + nrl_ids
        all_nrl_ids.append(nrl_ids)
    return all_nrl_ids


def pad_sequences(sequences, pad_id, device="cpu"):
    """Pad a list of variable-length sequences to form a batch tensor."""
    max_len = max(len(s) for s in sequences)
    padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for i, s in enumerate(sequences):
        padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attention_mask[i, :len(s)] = 1
    return padded, attention_mask


def decode_predictions_to_penman(preds, tokenizer, input_texts):
    """Decode model predictions (token ids) into PENMAN-formatted AMR strings."""
    penman_graphs = []
    for idx in range(len(preds)):
        ith_pred = list(preds[idx])
        # Replace special tokens
        if len(ith_pred) > 0:
            ith_pred[0] = tokenizer.bos_token_id
        ith_pred = [
            tokenizer.eos_token_id if itm == tokenizer.amr_eos_token_id else itm
            for itm in ith_pred if itm != tokenizer.pad_token_id
        ]

        try:
            graph, status, _ = tokenizer.decode_amr(ith_pred, restore_name_ops=False)
            graph.status = status
            metadata = {
                "id": str(idx),
                "annotator": "bart-amr",
                "snt": input_texts[idx] if idx < len(input_texts) else "",
            }
            graph.metadata = metadata

            # Fix empty concepts
            for i, triple in enumerate(graph.triples):
                if triple[1] == ":instance" and (triple[2] == "" or triple[2] is None):
                    graph.triples[i] = penman.Triple(triple[0], triple[1], "amr-unknown")

            graph = postprocessing.ensure_all_variables_have_instance(graph)
            txt = penman.encode(graph)
            txt = postprocessing.fix_unclosed_parentheses(txt)
            txt = postprocessing.fix_empty_concepts_in_amr_string(txt)
            txt = postprocessing.dedup_variables_in_amr_string(txt)
            penman_graphs.append(txt)
        except Exception as e:
            logger.warning(f"Failed to decode prediction {idx}: {e}")
            fallback = f'(z0 / thing\n    :wiki "-")'
            penman_graphs.append(fallback)

    return penman_graphs


def run_generation(model, tokenizer, text_ids, nrl_ids, device, batch_size=1,
                   num_beams=5, max_gen_length=512):
    """Run model.generate() with given text_ids and nrl_ids."""
    model.eval()
    all_predictions = []
    n = len(text_ids)
    
    for start in tqdm(range(0, n, batch_size), desc="Generating"):
        end = min(start + batch_size, n)
        batch_text = text_ids[start:end]
        batch_nrl = nrl_ids[start:end]

        input_ids, attention_mask = pad_sequences(batch_text, tokenizer.pad_token_id, device)
        nrl_input_ids, _ = pad_sequences(batch_nrl, tokenizer.pad_token_id, device)

        with torch.no_grad():
            generated = model.generate(
                input_ids,
                attention_mask=attention_mask,
                input_ids_dfs_NRL=nrl_input_ids,
                num_beams=num_beams,
                max_length=max_gen_length,
                decoder_start_token_id=tokenizer.amr_bos_token_id,
                eos_token_id=tokenizer.amr_eos_token_id,
                no_repeat_ngram_size=0,
                length_penalty=1.0,
                use_cache=True,
            )

        all_predictions.extend(generated.cpu().tolist())

    return all_predictions


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Inference for AMRBART-RGL")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Tokenizer name (defaults to model_name_or_path)")
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_source_length", type=int, default=400)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--generation_num_beams", type=int, default=5)
    parser.add_argument("--generation_max_length", type=int, default=512)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--unified_input", type=bool, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gold_amr_file", type=str, default=None,
                        help="Gold AMR file for smatch evaluation")
    parser.add_argument("--stage1_only", action="store_true",
                        help="Only run stage 1 (for comparison)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    logger.info(f"Loading tokenizer from: {tokenizer_name}")
    tokenizer = AMRBartTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=args.cache_dir,
        use_fast=False,
    )

    # Load model
    logger.info(f"Loading model from: {args.model_name_or_path}")
    model = BartForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    if args.fp16 and device.type == "cuda":
        model = model.half()
    model.eval()

    # Load test sentences
    logger.info(f"Loading test data from: {args.test_file}")
    sentences = load_test_sentences(args.test_file)
    logger.info(f"Loaded {len(sentences)} test sentences")

    # Prepare text inputs
    text_ids = prepare_text_input(
        tokenizer, sentences,
        max_length=args.max_source_length,
        unified_input=args.unified_input,
    )

    # ============================================================
    # STAGE 1: Generate rough AMR (text as NRL placeholder)
    # ============================================================
    logger.info("=" * 60)
    logger.info("STAGE 1: Generating rough AMR (text as NRL placeholder)")
    logger.info("=" * 60)

    # Use text_ids as NRL placeholder (same as current inference)
    stage1_nrl_ids = [list(ids) for ids in text_ids]

    stage1_preds = run_generation(
        model, tokenizer, text_ids, stage1_nrl_ids, device,
        batch_size=args.per_device_eval_batch_size,
        num_beams=args.generation_num_beams,
        max_gen_length=args.generation_max_length,
    )

    stage1_penman = decode_predictions_to_penman(stage1_preds, tokenizer, sentences)

    # Save stage 1 results
    stage1_file = os.path.join(args.output_dir, "stage1_predictions.txt")
    with open(stage1_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(stage1_penman))
    logger.info(f"Stage 1 results saved to: {stage1_file}")

    if args.stage1_only:
        logger.info("Stage 1 only mode. Done.")
        # Calculate smatch if gold file provided
        if args.gold_amr_file:
            _eval_smatch(args.gold_amr_file, stage1_file)
        return

    # ============================================================
    # CONVERT: Stage 1 AMR → NRL tokens
    # ============================================================
    logger.info("=" * 60)
    logger.info("CONVERTING: Stage 1 AMR → NRL tokens")
    logger.info("=" * 60)

    nrl_token_lists = []
    fallback_count = 0
    for idx, penman_str in enumerate(tqdm(stage1_penman, desc="AMR → NRL")):
        nrl_tokens = penman_to_nrl_tokens(penman_str)
        if nrl_tokens is None or len(nrl_tokens) < 3:
            # Fallback: use text as NRL (same as stage 1)
            nrl_token_lists.append(None)  # Mark for fallback
            fallback_count += 1
        else:
            nrl_token_lists.append(nrl_tokens)

    logger.info(f"Successfully converted {len(nrl_token_lists) - fallback_count}/{len(nrl_token_lists)} AMRs to NRL")
    if fallback_count > 0:
        logger.warning(f"{fallback_count} sentences will use text as NRL fallback")

    # Prepare NRL inputs
    stage2_nrl_ids = []
    for idx, nrl_tokens in enumerate(nrl_token_lists):
        if nrl_tokens is None:
            # Fallback to text input
            stage2_nrl_ids.append(list(text_ids[idx]))
        else:
            nrl_ids = tokenizer.tokenize_amr(nrl_tokens)[:args.max_source_length - 5]
            nrl_ids.append(tokenizer.amr_eos_token_id)
            if args.unified_input:
                nrl_ids = [
                    tokenizer.bos_token_id,
                    tokenizer.mask_token_id,
                    tokenizer.eos_token_id,
                    tokenizer.amr_bos_token_id,
                    tokenizer.amr_dfs_NRL_token_id,
                ] + nrl_ids
            stage2_nrl_ids.append(nrl_ids)

    # ============================================================
    # STAGE 2: Generate refined AMR (with real NRL from stage 1)
    # ============================================================
    logger.info("=" * 60)
    logger.info("STAGE 2: Generating refined AMR (with real NRL)")
    logger.info("=" * 60)

    stage2_preds = run_generation(
        model, tokenizer, text_ids, stage2_nrl_ids, device,
        batch_size=args.per_device_eval_batch_size,
        num_beams=args.generation_num_beams,
        max_gen_length=args.generation_max_length,
    )

    stage2_penman = decode_predictions_to_penman(stage2_preds, tokenizer, sentences)

    # Save stage 2 results
    stage2_file = os.path.join(args.output_dir, "stage2_predictions.txt")
    with open(stage2_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(stage2_penman))
    logger.info(f"Stage 2 results saved to: {stage2_file}")

    # Also save as the final output
    final_file = os.path.join(args.output_dir, "generated_predictions_penman.txt")
    with open(final_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(stage2_penman))
    logger.info(f"Final results saved to: {final_file}")

    # ============================================================
    # EVALUATE: Calculate smatch
    # ============================================================
    if args.gold_amr_file:
        logger.info("=" * 60)
        logger.info("EVALUATING: Calculating Smatch scores")
        logger.info("=" * 60)
        
        logger.info("--- Stage 1 (baseline) ---")
        _eval_smatch(args.gold_amr_file, stage1_file)
        
        logger.info("--- Stage 2 (two-stage) ---")
        _eval_smatch(args.gold_amr_file, stage2_file)


def _eval_smatch(gold_file, pred_file):
    """Calculate smatch score between gold and predicted AMR files."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from common.eval_smatch import calculate_smatch
        score = calculate_smatch(gold_file, pred_file)
        logger.info(f"Smatch: {score}")
        return score
    except Exception as e:
        logger.warning(f"Smatch evaluation failed: {e}")
        try:
            # Fallback: try amrlib
            import amrlib
            score = amrlib.evaluate.smatch_enhanced.compute_smatch(gold_file, pred_file)
            logger.info(f"Smatch (amrlib): {score}")
            return score
        except Exception as e2:
            logger.warning(f"amrlib smatch also failed: {e2}")
            return None


if __name__ == "__main__":
    main()
