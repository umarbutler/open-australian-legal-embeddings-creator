from typing import Callable

import semchunk
from transformers import AutoTokenizer

HEADER = """\
Title: {citation}
Jurisdiction: {jurisdiction}
Type: {type}
"""
"""A template header to be added to the start of every chunk to provide additional context to the embedding model."""

JURISDICTIONS = {
    'commonwealth' : 'Commonwealth of Australia',
    'new_south_wales' : 'New South Wales',
    'norfolk_island' : 'Norfolk Island',
    'queensland' : 'Queensland',
    'south_australia' : 'South Australia',
    'tasmania' : 'Tasmania',
    'western_australia' : 'Western Australia',
}
"""A map of jurisdictions to the name to be used in the chunk header."""

TYPES = {
    'primary_legislation' : 'Act',
    'secondary_legislation' : 'Regulation',
    'bill' : 'Bill',
    'decision' : 'Judgment',
}
"""A map of document types to the name to be used in the chunk header."""

def get_chunker(model_name: str, chunk_size: int) -> Callable[[dict[str, str]], tuple[list[dict[str, str]], list[str]]]:
    """Generate a chunker for the provided Hugging Face Transformers model and chunk size."""
    
    # Generate a token counter for the model.
    tokeniser = AutoTokenizer.from_pretrained(model_name, model_max_length=None)

    def token_counter(text: str) -> int:
        """Count the number of tokens in the provided text."""
        
        return tokeniser(text, return_length=True)['length'][0]
    
    # Generate the chunker.
    def chunker(doc: dict[str, str]) -> tuple[list[dict[str, str]], list[str], list[int]]:
        """Split a document into chunks of the provided size."""
        
        # Create a header to append to every chunk to provide additional context to the embedding model.
        header = HEADER.format(
            citation = doc['citation'],
            jurisdiction = JURISDICTIONS[doc['jurisdiction']],
            type = TYPES[doc['type']]
        )
        
        # Split the document into chunks of the provided size.
        chunks = [header + chunk for chunk in semchunk.chunk(doc['text'], chunk_size - token_counter(header), token_counter)]
        
        # Return empty lists if there are no chunks.
        if not chunks:
            return [], [], []
        
        # Remove the document's text from its metadata.
        del doc['text']
        
        # Initialise a flag for whether a chunk is the last chunk.
        doc['is_last_chunk'] = False
        
        # Create metadata for every chunk.
        chunks_len = len(chunks)
        metadatas = [doc] * chunks_len
        metadatas[-1]['is_last_chunk'] = True
        
        return chunks, metadatas, [len(header)] * chunks_len

    return chunker