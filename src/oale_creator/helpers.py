import itertools
import os
from collections.abc import Generator, Iterable
from typing import Any

import orjson
from rich.console import Console

console = Console()

def save_json(path: str, content: Any) -> None:
    """Save content as a json file."""
    
    with open(path, 'wb') as writer:
        writer.write(orjson.dumps(content))

def load_json(path: str) -> Any:
    """Load a json file."""
    
    with open(path, 'rb') as reader:
        return orjson.loads(reader.read())

def count_lines(path: str) -> int:
    """Count the number of lines in a file."""
    
    with open(path, 'rb') as file:
        return sum(1 for _ in file)

def remove_lines(path: str, indices: set[int]) -> None:
    """Remove lines from the provided file at the specified indices."""
    
    # Create a temporary copy of the file excluding lines at the specified indices.
    with open(path, 'rb') as file, open(f'{path}.tmp', 'wb') as tmp_file:
        for i, line in enumerate(file):
            if i not in indices:
                tmp_file.write(line)
    
    # Overwrite the file with the filtered copy.
    os.replace(f'{path}.tmp', path)

def batch_generator(iterable: Iterable, batch_size: int) -> Generator[list, None, None]:
    """Generate batches of the specified size from the provided iterable."""
    
    iterator = iter(iterable)
    
    for first in iterator:
        yield list(itertools.chain([first], itertools.islice(iterator, batch_size - 1)))