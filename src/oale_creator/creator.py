import itertools
import os
import shutil

import mpire
import orjson
import orjsonl
from rich.markdown import Markdown
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .chunking import get_chunker
from .helpers import (batch_generator, console, count_lines, load_json,
                      remove_lines, save_json)
from .metadata import DATA_VERSION


class Creator:
    """The creator of the Open Australian Legal Embeddings."""
    
    def __init__(self,
                 corpus_path: str = 'corpus.jsonl',
                 data_dir: str = 'data',
                 model_name: str = 'BAAI/bge-small-en-v1.5',
                 chunk_size: int = 512,
                 chunking_batch_size: int = 4096,
                 embedding_batch_size: int = 32,
                 ) -> None:
        
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunking_batch_size = chunking_batch_size
        self.embedding_batch_size = embedding_batch_size
        
        # Retrieve the current working directory.
        cwd = os.getcwd()

        # Retrieve the absolute path to the Corpus by joining it with the current working directory.
        self.corpus_path = os.path.join(cwd, corpus_path)
        
        # Raise an error if the Corpus does not exist.
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f'Open Australian Legal Corpus not found at {self.corpus_path}.')
        
        # Retrieve the absolute path to the data directory by joining it with the current working directory.
        data_dir = os.path.join(cwd, data_dir)
        
        # Create paths to data files by joining them with the data directory.
        self.embeddings_path = os.path.join(data_dir, 'embeddings.jsonl')
        self.metadatas_path = os.path.join(data_dir, 'metadatas.jsonl')
        self.texts_path = os.path.join(data_dir, 'texts.jsonl')
        config_path = os.path.join(data_dir, 'version.json')
        
        # Freeze the current configuration of the Creator.
        config = {
            'data_version' : DATA_VERSION,
            'model' : self.model_name,
            'chunk_size' : self.chunk_size,
        }
        
        # Delete any incompatible data.
        if os.path.exists(config_path) and load_json(config_path) != config:
            shutil.rmtree(data_dir)
        
        # Create any necessary directories.
        os.makedirs(data_dir, exist_ok=True)
        
        # Create data files if they don't exist.
        for file in {self.embeddings_path, self.metadatas_path, self.texts_path}:
            if not os.path.exists(file):
                with open(file, 'wb') as _: pass
        
        # Store the current configuration of the Creator.
        save_json(config_path, config)
        
    def create(self) -> None:
        """Update the Embeddings."""
        
        console.print(Markdown('# Open Australian Legal Embeddings Creator'), style='light_cyan1')
        
        # Identify documents that have already been embedded and search for corrupted documents to remove.
        total_metadatas = count_lines(self.metadatas_path)
        existing_version_ids = []
        remove_indices = set()
        
        if total_metadatas:
            console.print('Identifying documents that have already been embedded and searching for corrupted documents to remove.', style='light_cyan1 bold')
            last_first_chunk_version_id = None
            last_first_chunk_i = 0
            
            for i, metadata in enumerate(tqdm(orjsonl.stream(self.metadatas_path), total=total_metadatas, unit=' chunk')):
                version_id = metadata['version_id']
                
                if version_id != last_first_chunk_version_id:
                    last_first_chunk_version_id = version_id
                    last_first_chunk_i = i
                
                existing_version_ids.append(version_id)
                
                # If the last document chunked is missing chunks, add the indices of its chunks to `remove_indices` and remove its version id from `existing_version_ids`.
                if i+1 == total_metadatas and not metadata['is_last_chunk']:
                    remove_indices.update(range(last_first_chunk_i, i+1))
                    
                    for i in reversed(range(len(existing_version_ids))):
                        if existing_version_ids[i] != last_first_chunk_version_id:
                            break
                        
                        del existing_version_ids[i]
        
        existing_version_ids_set = set(existing_version_ids)

        # Add the indices of data missing from any one of the data files to `remove_indices`.
        lines = [count_lines(path) for path in {self.embeddings_path, self.texts_path}] + [total_metadatas]
        remove_indices.update(range(min(lines), max(lines)))
                
        # Identify outdated documents to remove and missing documents to embed.
        console.print('\nIdentifying outdated documents to remove and missing documents to embed.', style='light_cyan1 bold')
        version_ids = [doc['version_id'] for doc in tqdm(orjsonl.stream(self.corpus_path), total=count_lines(self.corpus_path), unit=' document')]
        version_ids_set = set(version_ids)
        
        missing_indices = {i for i, version_id in enumerate(version_ids) if version_id not in existing_version_ids_set}
        remove_indices.update(i for i, version_id in enumerate(existing_version_ids) if version_id not in version_ids_set)
        
        # Remove outdated and corrupted documents.
        if remove_indices:
            console.print('\nRemoving outdated and corrupted documents.', style='light_cyan1 bold')
            
            with mpire.WorkerPool() as pool:
                pool.map_unordered(remove_lines, list(itertools.product({self.embeddings_path, self.metadatas_path, self.texts_path}, [remove_indices])), progress_bar=True, progress_bar_options={'unit': ' file'})
        
        # Return if there are no missing documents.
        if not missing_indices:
            console.print('\nThe Embeddings are already up to date.', style='dark_cyan bold')
            return
        
        # Create a chunker for the provided model and chunk size.
        chunker = get_chunker(self.model_name, self.chunk_size)
        
        # Load the Sentence Transformer model.
        model = SentenceTransformer(self.model_name, device='cuda')
        
        # Embed missing documents.
        console.print('\nEmbedding missing documents.', style='light_cyan1 bold')
        with open(self.corpus_path, 'rb') as corpus_file, \
            open(self.embeddings_path, 'ab') as embeddings_file, \
            open(self.metadatas_path, 'ab') as metadatas_file, \
            open(self.texts_path, 'ab') as texts_file, \
            tqdm(total=len(missing_indices), unit=' document') as bar, \
            mpire.WorkerPool(use_dill=True) as pool:
                doc_i = -1

                for docs_batch in batch_generator(corpus_file, self.chunking_batch_size):
                    # Filter out documents that have already been embedded, and deserialise those that have not.
                    batch_i = -1
                    
                    for _ in range(len(docs_batch)):
                        batch_i += 1
                        doc_i += 1
                        
                        if doc_i not in missing_indices:
                            del docs_batch[batch_i]
                            batch_i -= 1
                        
                        else:
                            docs_batch[batch_i] = {'doc': orjson.loads(docs_batch[batch_i])}
                    
                    # Continue if there are no documents to embed.
                    if not docs_batch:
                        continue

                    # Chunk the documents.                   
                    for embeddings_batch in batch_generator(pool.imap_unordered(chunker, docs_batch) if len(docs_batch) == self.chunking_batch_size else [chunker(**doc) for doc in docs_batch], self.embedding_batch_size):
                        chunks_metadatas_header_lengths = zip(*embeddings_batch)

                        # NOTE It is possible for no chunks to be returned if all the documents in the batch are empty.
                        if chunks_metadatas_header_lengths:
                            chunks, metadatas, header_lengths = chunks_metadatas_header_lengths
                            
                            # Flatten the chunks, metadatas and header lengths.
                            chunks = [chunk for chunks in chunks for chunk in chunks]
                            metadatas = [metadata for metadatas in metadatas for metadata in metadatas]
                            header_lengths = [header_length for header_lengths in header_lengths for header_length in header_lengths]
                            
                            # Embed the chunks.
                            embeddings = model.encode(chunks, batch_size=self.embedding_batch_size, normalize_embeddings=True)
                            
                            # Serialise the embeddings as json.
                            embeddings = [orjson.dumps(embedding, option=orjson.OPT_SERIALIZE_NUMPY) for embedding in embeddings]
                            
                            # Remove headers from the chunks and serialise the resulting texts as json.
                            chunks = [orjson.dumps(chunk[header_length:]) for chunk, header_length in zip(chunks, header_lengths)]
                            
                            # Serialise the metadatas as json.
                            metadatas = [orjson.dumps(metadata) for metadata in metadatas]
                                                
                            # Save the embeddings, metadatas and texts.
                            for embedding, metadata, text in zip(embeddings, metadatas, chunks):
                                embeddings_file.write(embedding)
                                embeddings_file.write(b'\n')
                                
                                metadatas_file.write(metadata)
                                metadatas_file.write(b'\n')
                                
                                texts_file.write(text)
                                texts_file.write(b'\n')
                        
                        bar.update(len(embeddings_batch))

        console.print('\nThe embeddings have been updated!', style='dark_cyan bold')