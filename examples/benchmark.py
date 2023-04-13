import click
import time
from tqdm import tqdm


@click.command()
@click.option("--model_name", default="THUDM/chatglm-6b")
@click.option(
    "--prompt",
    default="Human: Write a paragraph about AI, requiring no less than 800 words \n\nAssistant: ",
)
@click.option("--warmup_iters", default=1)
@click.option("--iters", default=20)
@click.option("--new_tokens", default=128)
@click.option("--fast/--no-fast", default=True)
@click.option("--quant/--no-quant", default=False)
def benchmark(model_name, prompt, warmup_iters, iters, new_tokens, fast, quant):
    if fast:
        import faster_chatglm_6b
    from transformers import AutoModel, AutoTokenizer
    import torch

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if quant:
            model.quantize(bits=8)
        model.half().cuda()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        def generate_fn():
            generate_ids = model.generate(
                input_ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
            )
            generate_ids.cpu().numpy()
            return generate_ids

        for i in range(warmup_iters):
            generate_fn()

        t0 = time.time()
        for r in tqdm(range(iters)):
            generate_ids = generate_fn()
        t1 = time.time()
        duration = t1 - t0
        average = duration / iters
        output = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        response = output[len(prompt) :]
        print(response)
        print(
            f"Finish {iters} iters in {duration:.3f} seconds, average {average:.2f}s/it"
        )


if __name__ == "__main__":
    benchmark()
