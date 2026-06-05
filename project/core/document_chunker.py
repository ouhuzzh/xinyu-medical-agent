import os
import glob
import re
import config
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class DocumentChuncker:
    def __init__(self):
        self.__parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.HEADERS_TO_SPLIT_ON, 
            strip_headers=False
        )
        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE, 
            chunk_overlap=config.CHILD_CHUNK_OVERLAP
        )
        self.__min_parent_size = config.MIN_PARENT_SIZE
        self.__max_parent_size = config.MAX_PARENT_SIZE

    def create_chunks(self, path_dir=config.MARKDOWN_DIR):
        all_parent_chunks, all_child_chunks = [], []

        for doc_path_str in sorted(glob.glob(os.path.join(path_dir, "*.md"))):
            doc_path = Path(doc_path_str)
            parent_chunks, child_chunks = self.create_chunks_single(doc_path)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)
        
        return all_parent_chunks, all_child_chunks

    def create_chunks_single(self, md_path):
        doc_path = Path(md_path)

        with open(doc_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        metadata = self.__extract_front_matter_metadata(raw_text)
        content_text = self.__strip_front_matter(raw_text)
        parent_chunks = self.__parent_splitter.split_text(content_text)
        
        merged_parents = self.__merge_small_parents(parent_chunks)
        split_parents = self.__split_large_parents(merged_parents)
        cleaned_parents = self.__clean_small_chunks(split_parents)
        
        all_parent_chunks, all_child_chunks = [], []
        self.__create_child_chunks(all_parent_chunks, all_child_chunks, cleaned_parents, doc_path, metadata)
        return all_parent_chunks, all_child_chunks

    @staticmethod
    def __extract_front_matter_metadata(raw_text):
        lines = raw_text.splitlines()
        metadata = {}
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if metadata:
                    break
                continue
            match = re.match(r"^([A-Za-z][A-Za-z0-9 _-]*):\s*(.+?)\s*$", stripped)
            if not match:
                if metadata:
                    break
                continue
            key = match.group(1).strip().lower().replace(" ", "_")
            metadata[key] = match.group(2).strip()
        return metadata

    @staticmethod
    def __strip_front_matter(raw_text):
        lines = raw_text.splitlines()
        stripped_lines = []
        seen_metadata = False
        metadata_done = False
        for line in lines:
            stripped = line.strip()
            is_metadata_line = bool(re.match(r"^([A-Za-z][A-Za-z0-9 _-]*):\s*(.+?)\s*$", stripped))
            if not metadata_done and is_metadata_line:
                seen_metadata = True
                continue
            if seen_metadata and not metadata_done:
                if stripped:
                    metadata_done = True
                    stripped_lines.append(line)
                continue
            stripped_lines.append(line)
        return "\n".join(stripped_lines)

    def __merge_small_parents(self, chunks):
        if not chunks:
            return []
        
        merged, current = [], None
        
        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in current.metadata:
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"
                    else:
                        current.metadata[k] = v

            if len(current.page_content) >= self.__min_parent_size:
                merged.append(current)
                current = None
        
        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    if k in merged[-1].metadata:
                        merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                    else:
                        merged[-1].metadata[k] = v
            else:
                merged.append(current)
        
        return merged

    def __split_large_parents(self, chunks):
        split_chunks = []
        
        for chunk in chunks:
            if len(chunk.page_content) <= self.__max_parent_size:
                split_chunks.append(chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.__max_parent_size,
                    chunk_overlap=config.CHILD_CHUNK_OVERLAP
                )
                sub_chunks = splitter.split_documents([chunk])
                split_chunks.extend(sub_chunks)
        
        return split_chunks

    def __clean_small_chunks(self, chunks):
        cleaned = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self.__min_parent_size:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata:
                            cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                        else:
                            cleaned[-1].metadata[k] = v
                elif i < len(chunks) - 1:
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    for k, v in chunk.metadata.items():
                        if k in chunks[i + 1].metadata:
                            chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                        else:
                            chunks[i + 1].metadata[k] = v
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)
        
        return cleaned

    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path, base_metadata):
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"{doc_path.stem}_parent_{i}"
            section_title = (
                p_chunk.metadata.get("H3")
                or p_chunk.metadata.get("H2")
                or p_chunk.metadata.get("H1")
                or base_metadata.get("title")
                or doc_path.stem
            )
            p_chunk.metadata.update(base_metadata)
            p_chunk.metadata.update(
                {
                    "source": doc_path.name,
                    "parent_id": parent_id,
                    "section_title": section_title,
                    "document_topic": base_metadata.get("title") or doc_path.stem,
                    "intended_audience": base_metadata.get("source_type") or base_metadata.get("audience") or "general",
                    "source_version": base_metadata.get("version") or base_metadata.get("published_at") or "",
                }
            )
            
            all_parent_pairs.append((parent_id, p_chunk))
            all_child_chunks.extend(self.__child_splitter.split_documents([p_chunk]))
