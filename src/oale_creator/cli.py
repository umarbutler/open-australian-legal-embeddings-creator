import os

import click
import rich
from rich.traceback import install

from .creator import Creator

# Set up traceback pretty printing with `rich` (suppressing full traceback for exceptions raised by `rich` and `click`).
install(suppress=[rich, click])

@click.command('mkoale', context_settings={'help_option_names': ['-h', '--help']})
@click.version_option()
@click.option(
    '-i', '--input',
    default=os.path.join(os.getcwd(), 'corpus.jsonl'),
    show_default=True,
    help='The path to the Open Australian Legal Corpus.',
)
@click.option(
    '-o', '--output',
    default=os.path.join(os.getcwd(), 'data'),
    show_default=True,
    help='The directory in which the Embeddings should be stored.',
)
@click.option(
    '-m', '--model',
    default='BAAI/bge-small-en-v1.5',
    show_default=True,
    help='The name of the Hugging Face Sentence Transformer embedding model to use.',
)
@click.option(
    '-c', '--chunk_size',
    default=512,
    show_default=True,
    help='The maximum number of tokens a chunk may contain.',
)
@click.option(
    '-cb', '--chunking_batch_size',
    default=4096,
    show_default=True,
    help='The maximum number of documents that may be chunked at once.',
)
@click.option(
    '-eb', '--embedding_batch_size',
    default=32,
    show_default=True,
    help='The maximum number of chunks that may be embedded at once.',
)
def create(input, output, model, chunk_size, chunking_batch_size, embedding_batch_size):
    """The creator of the Open Australian Legal Embeddings."""
    
    Creator(
        corpus_path=input,
        data_dir=output,
        model_name=model,
        chunk_size=chunk_size,
        chunking_batch_size=chunking_batch_size,
        embedding_batch_size=embedding_batch_size,
    ).create()

if __name__ == '__main__':
    create()