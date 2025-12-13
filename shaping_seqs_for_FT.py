def group_sequences(examples, block_size):
    """
    Concatenate tokenized sequences and split them into fixed-size blocks for model training.
    """
    concatenated_sequences = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_sequences[list(examples.keys())[0]])

    if total_length >= block_size:
        # NOTE: it might look like this does nothing. What it's actually doing is:
        # 1. Find out the total number of blocks that fit the total length of the sequences
        # 2. Drops the remainder by casting into an int
        # 3. Multiplies by `block_size` to find how long the total sequences will be when split into blocks
        total_length = (total_length // block_size) * block_size

    result = {
        k: [seq[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, seq in concatenated_sequences.items()
    }
    result["labels"] = result["input_ids"].copy()  # Make a shallow copy
    return result

