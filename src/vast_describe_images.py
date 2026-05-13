"""Run VLM image descriptions on a vast.ai GPU.

Supports multiple model backends; pick via --model.
Saves per-model JSON cache: image_hash -> description.
Resumes on re-run.

Usage on vast:
  python3 vast_describe_images.py --model qwen2vl
  python3 vast_describe_images.py --model llama-vision
  python3 vast_describe_images.py --model llava
  python3 vast_describe_images.py --model blip2

Each run produces a separate cache file: descriptions_{model}.json
"""
import argparse, json, os, time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROMPT_ES = (
    "Describe esta imagen de noticia política en español. Sé específico: "
    "menciona personas concretas con nombre si las reconoces, lugares, "
    "edificios institucionales, símbolos, banderas, y cualquier objeto o "
    "texto destacado. 2-3 frases."
)


def load_qwen2vl(model_id="Qwen/Qwen2-VL-7B-Instruct"):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print(f"Loading {model_id}...", flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    def gen(image: Image.Image, prompt: str) -> str:
        messages = [{"role":"user","content":[
            {"type":"image","image":image},
            {"type":"text","text":prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        gen_ids = out[:, inputs.input_ids.shape[1]:]
        text_out = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return text_out.strip()
    return gen


def load_llama_vision(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    print(f"Loading {model_id}...", flush=True)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    def gen(image: Image.Image, prompt: str) -> str:
        messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        text_out = processor.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return text_out.strip()
    return gen


def load_llava(model_id="llava-hf/llava-1.5-7b-hf"):
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    print(f"Loading {model_id}...", flush=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    def gen(image: Image.Image, prompt: str) -> str:
        conversation = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        text_out = processor.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return text_out.strip()
    return gen


def load_blip2(model_id="Salesforce/blip2-flan-t5-xl"):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    print(f"Loading {model_id}...", flush=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = Blip2Processor.from_pretrained(model_id)
    def gen(image: Image.Image, prompt: str) -> str:
        # BLIP-2 takes a free-form question+image
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.bfloat16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        return processor.decode(out[0], skip_special_tokens=True).strip()
    return gen


LOADERS = {
    "qwen2vl": (load_qwen2vl, "Qwen/Qwen2-VL-7B-Instruct"),
    "qwen2vl-2b": (load_qwen2vl, "Qwen/Qwen2-VL-2B-Instruct"),
    "llama-vision": (load_llama_vision, "meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "llava": (load_llava, "llava-hf/llava-1.5-7b-hf"),
    "blip2": (load_blip2, "Salesforce/blip2-flan-t5-xl"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache_path", default=None)
    args = ap.parse_args()

    loader_fn, model_id = LOADERS[args.model]
    cache_path = Path(args.cache_path or RESULTS_DIR / f"descriptions_{args.model}.json")

    test = pd.read_csv(DATA_DIR / "test_public" / "test_public.csv", encoding="utf-8")
    hashes = list(test["image_hash"].unique())
    if args.limit:
        hashes = hashes[: args.limit]
    print(f"Total hashes: {len(hashes)}")

    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"Resuming from cache: {len(cache)} done")

    todo = [h for h in hashes if h not in cache]
    print(f"To do: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    gen = loader_fn(model_id)
    img_dir = DATA_DIR / "test_public" / "images"

    t0 = time.time()
    failed = 0
    for i, h in enumerate(todo, start=1):
        img_path = img_dir / f"{h}.jpg"
        try:
            img = Image.open(img_path).convert("RGB")
            desc = gen(img, PROMPT_ES)
        except Exception as e:
            desc = None
        if desc:
            cache[h] = desc
        else:
            failed += 1
        if i % 50 == 0 or i == len(todo):
            rate = i / max(1, time.time() - t0)
            eta = (len(todo) - i) / max(1e-3, rate)
            print(f"  {i}/{len(todo)}  {rate:.2f}/s  eta={eta:.0f}s  failed={failed}", flush=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(f"\nFinished. Cache size: {len(cache)}.  Failed: {failed}.  Saved to {cache_path}")


if __name__ == "__main__":
    main()
