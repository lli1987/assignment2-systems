local_benchmark_config = {
    "name": "benchmark",
    "input_file": "/Users/luyaoli/code/cs336/assignment2-systems/cs336_systems/data/TinyStoriesV2-GPT4-train.txt",
    "encoding_output_dir": "/Users/luyaoli/code/cs336/assignment2-systems/cs336_systems/encoding",
    "serde_output_dir": "/Users/luyaoli/code/cs336/assignment2-systems/cs336_systems/serde",
    "is_local": True,
    "training_steps": 100,
    "warmup_steps": 0,
    "device": "cpu",
    "train_bpe": False,
    "encode_ids": False,
    "use_amp": False,
}

benchmark_config = {
    "name": "benchmark",
    "input_file": "/workspace/data/TinyStoriesV2-GPT4-train.txt",
    "encoding_output_dir": "/workspace/code/assignment2-systems/cs336_systems/encoding",
    "serde_output_dir": "/workspace/code/assignment2-systems/cs336_systems/serde",
    "is_local": False,
    "training_steps": 500,
    "warmup_steps": 20,
    "device": "cuda",
    "train_bpe": False,
    "encode_ids": True,
    "use_amp": True,
}
