import os
import string

# ---------- helpers ----------

def chunk_label(n: int) -> str:
    """0->A, 25->Z, 26->AA, etc."""
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def utf8_size(s: str) -> int:
    return len(s.encode("utf-8"))

def bytes_human(n: int) -> str:
    # human-ish readout for logs
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n/1024:.1f} {unit}"
        n /= 1024

# ---------- core ----------

def find_md_files(directory):
    """Find all .md files in the given directory and its subdirectories."""
    md_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".md"):
                md_files.append(os.path.join(root, file))
    # deterministic order
    return sorted(md_files)

def open_new_chunk(output_file_base, chunk_index):
    label = chunk_label(chunk_index)
    path = f"{output_file_base}_{label}.md"
    f = open(path, "w", encoding="utf-8")
    print(f"Creating chunk file: {path}")
    return f, path

def concatenate_files_by_size(
    file_paths,
    output_file_base,
    max_chunk_bytes=500 * 1024 * 1024,  # ~500 MB safe target
):
    """
    Concatenate all files with path headers into multiple output files,
    rolling to a new chunk when the next write would exceed max_chunk_bytes.
    """
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")

    # constants
    sep = "\n\n---\n"
    sep_sz = utf8_size(sep)

    chunk_index = 0
    chunk_bytes = 0
    chunks = []
    outfile, outpath = open_new_chunk(output_file_base, chunk_index)
    chunks.append(outpath)

    cwd = os.getcwd()

    for file_path in file_paths:
        relative_path = os.path.relpath(file_path, cwd)

        # header before a new file's content
        header = f"\n\n## File: {relative_path}\n\n"
        header_sz = utf8_size(header)

        # If header won't fit, roll to next chunk (unless we're at empty chunk, then it must fit as first write)
        if chunk_bytes > 0 and (chunk_bytes + header_sz > max_chunk_bytes):
            outfile.write(sep)
            chunk_bytes += sep_sz
            outfile.close()

            chunk_index += 1
            outfile, outpath = open_new_chunk(output_file_base, chunk_index)
            chunks.append(outpath)
            chunk_bytes = 0

        outfile.write(header)
        chunk_bytes += header_sz

        # stream file contents line-by-line
        in_mid_file = False
        try:
            with open(file_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    line_sz = utf8_size(line)

                    # If this line would overflow the chunk, roll and add continuation header
                    if chunk_bytes > 0 and (chunk_bytes + line_sz > max_chunk_bytes):
                        # end current chunk neatly
                        if chunk_bytes + sep_sz <= max_chunk_bytes:
                            outfile.write(sep)
                            chunk_bytes += sep_sz
                        outfile.close()

                        chunk_index += 1
                        outfile, outpath = open_new_chunk(output_file_base, chunk_index)
                        chunks.append(outpath)
                        chunk_bytes = 0

                        # continuation header
                        cont = f"\n\n## File (continued): {relative_path}\n\n"
                        cont_sz = utf8_size(cont)
                        # If the continuation header alone exceeds max size (path too long / tiny size),
                        # still write it (first write in a new chunk), then proceed.
                        outfile.write(cont)
                        chunk_bytes += cont_sz
                        in_mid_file = True

                    outfile.write(line)
                    chunk_bytes += line_sz

        except Exception as e:
            err = f"Error reading file {relative_path}: {str(e)}\n"
            err_sz = utf8_size(err)
            if chunk_bytes > 0 and (chunk_bytes + err_sz > max_chunk_bytes):
                # finish chunk
                if chunk_bytes + sep_sz <= max_chunk_bytes:
                    outfile.write(sep)
                    chunk_bytes += sep_sz
                outfile.close()

                chunk_index += 1
                outfile, outpath = open_new_chunk(output_file_base, chunk_index)
                chunks.append(outpath)
                chunk_bytes = 0

            outfile.write(err)
            chunk_bytes += err_sz

        # write separator after finishing each file (if it fits; else roll first)
        if chunk_bytes > 0 and (chunk_bytes + sep_sz > max_chunk_bytes):
            outfile.close()
            chunk_index += 1
            outfile, outpath = open_new_chunk(output_file_base, chunk_index)
            chunks.append(outpath)
            chunk_bytes = 0

        outfile.write(sep)
        chunk_bytes += sep_sz

    outfile.close()
    return chunks

def main():
    current_dir = os.getcwd()
    output_file_base = os.path.join(current_dir, "All")

    print(f"Searching for .md files in {current_dir} and subdirectories...")
    md_files = find_md_files(current_dir)

    if not md_files:
        print("No .md files found.")
        return

    print(f"Found {len(md_files)} .md files. Concatenating into ~500 MB chunks...")
    chunks = concatenate_files_by_size(
        md_files,
        output_file_base,
        max_chunk_bytes=500 * 1024 * 1024,  # tweak if you want a different target
    )

    # final log
    first = os.path.basename(chunks[0]) if chunks else "N/A"
    last = os.path.basename(chunks[-1]) if chunks else "N/A"
    print(f"Concatenation complete. Created {len(chunks)} chunk files ({first} through {last}).")

if __name__ == "__main__":
    main()
