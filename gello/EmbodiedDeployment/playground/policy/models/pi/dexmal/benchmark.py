import time

import torch
from pi0_infer import Pi0Inference


def debugpy_listen():
    import debugpy

    debugpy.listen(("0.0.0.0", 10092))
    print("Waiting for client to attach 10092...")
    debugpy.wait_for_client()


if __name__ == "__main__":
    import argparse

    import numpy as np

    np.random.seed(100)
    torch.manual_seed(100)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_views", type=int, default=3, help="Number of views"
    )
    parser.add_argument(
        "--prompt_len", type=int, default=10, help="Prompt length"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=50, help="Chunk size"
    )
    args = parser.parse_args()

    # debugpy_listen()

    infer = Pi0Inference(
        {
            "language_embeds": torch.randn(
                args.prompt_len, 2048, dtype=torch.bfloat16
            ),  # 空的
        },
        num_views=args.num_views,
        chunk_size=args.chunk_size,
    )

    input_image = torch.randn(
        args.num_views, 224, 224, 3, dtype=torch.bfloat16
    ).cuda()
    input_state = torch.randn(32, dtype=torch.bfloat16).cuda()
    input_noise = torch.randn(args.chunk_size, 32, dtype=torch.bfloat16).cuda()

    # Warm up
    for _ in range(3):
        _ = infer.forward(input_image, input_state, input_noise)
        torch.cuda.synchronize()

    # Benchmark
    iterations = 100
    times = []
    for _ in range(iterations):
        t0 = time.time()
        _ = infer.forward(input_image, input_state, input_noise)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
    print(
        "views",
        args.num_views,
        "prompt_len",
        args.prompt_len,
        "chunk_size",
        args.chunk_size,
    )
    print(
        "runs",
        len(times),
        "median time per inference:",
        "%.3f" % (sorted(times)[len(times) // 2] * 1000),
        "ms",
    )
