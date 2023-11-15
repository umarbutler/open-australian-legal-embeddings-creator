# Open Australian Legal Embeddings Creator
The [Open Australian Legal Embeddings](https://huggingface.co/datasets/umarbutler/open-australian-legal-embeddings) are the first open-source embeddings of Australian legislative and judicial documents. This repository contains the code used to create and update the Embeddings.

If you're looking to download the Embeddings, you may do so on [Hugging Face](https://huggingface.co/datasets/umarbutler/open-australian-legal-embeddings).

## Installation
To install the Creator, run the following commands:
```bash
git clone https://github.com/umarbutler/open-australian-legal-embeddings-creator.git
cd open-australian-legal-embeddings-creator
pip install .
```

## Usage
To create or update the Embeddings, simply call `mkoale` from the directory in which the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus) is located. By default, this will output the Embeddings to a folder named `data` in the current working directory.

The Creator's default behaviour may be modified by passing the following optional arguments to `mkoale`:
* `-i`/`--input`: The path to the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus). Defaults to a file named `corpus.jsonl` in the current working directory.
* `-o`/`--output`: The directory in which the Embeddings should be stored. Defaults to a folder named `data` in the current working directory.
* `-m`/`--model`: The name of the Hugging Face Sentence Transformer embedding model to use. Defaults to [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5).
* `-c`/`--chunk_size`: The maximum number of tokens a chunk may contain. Defaults to 512.
* `-cb`/`--chunking_batch_size`: The maximum number of documents that may be chunked at once. Defaults to 4096.
* `-em`/`--embedding_batch_size`: The maximum number of chunks that may be embedded at once. Defaults to 32.

## Licence
The Creator is licensed under the [MIT License](LICENCE).