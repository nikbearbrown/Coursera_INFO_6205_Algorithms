
# Github Markdown collector

It scans the current directory (and all subfolders) for **.md files**, then **concatenates** them into a series of Markdown “chunk” files named like `All_A.md`, `All_B.md`, … Each chunk is kept under a **target size** (default ~500 MB). Between files it inserts a separator and a header showing the source path. If a chunk would overflow, it “rolls over” to a new chunk and continues.

# How it works (flow)

1. **Discovery**

   * `find_md_files(directory)` walks the tree and returns a **sorted** list of `.md` paths (deterministic order).

2. **Chunk setup**

   * `chunk_label(n)` maps `0→A, …, 25→Z, 26→AA, …` for filenames.
   * `open_new_chunk(base, i)` creates `base_<Label>.md` (e.g., `All_A.md`) and logs it.

3. **Concatenation with size budgeting**

   * Uses **UTF-8 byte size** (`utf8_size`) to count exactly what will be written.
   * For each source file:

     * Writes a header:
       `## File: <relative/path/from/cwd>`
     * Streams the file **line-by-line**:

       * If the **next line** would push the chunk over `max_chunk_bytes`, it:

         * Optionally closes with a separator `\n\n---\n` (if it fits),
         * Opens a new chunk,
         * Writes a **continuation header**:
           `## File (continued): <relative/path>`
         * Continues streaming lines.
     * After finishing a file, writes the separator. If the separator doesn’t fit, it rolls to a new chunk first.

4. **Errors while reading a file**

   * Catches exceptions, writes an **error message** into the output (rolling chunks if needed), then continues.

5. **Completion**

   * Closes the last chunk and prints a summary like:
     `Concatenation complete. Created N chunk files (All_A.md through All_<...>.md).`

# What it outputs

* Multiple Markdown files in the **current working directory**, named `All_A.md`, `All_B.md`, … each ~≤ 500 MB (target), containing:

  * Repeated blocks of:

    * `## File: <relative path>` (or `File (continued)` if a file spans chunks)
    * The original file’s content
    * A separator line `---`

# Notable behaviors & edge cases

* **Deterministic order:** files are sorted lexicographically by full path.
* **Accurate sizing:** counts **UTF-8 bytes**, not characters.
* **Very long lines:** if a **single line** exceeds `max_chunk_bytes`, it will be written at the start of a new chunk and that chunk may exceed the limit (the overflow check only triggers when `chunk_bytes > 0`).
* **Tiny chunk sizes:** if `max_chunk_bytes` is set extremely small (smaller than headers), the code still writes headers at the start of a new chunk (by design), so chunks may slightly exceed the limit.
* **Platform:** reads text as UTF-8; non-UTF-8 files will trigger the error path and log into the output.

# How to use it

* Save as `concat_md.py` in the directory whose `.md` files you want to combine.
* Run:

  ```bash
  python concat_md.py
  ```
* Optional: change the target chunk size by editing the `max_chunk_bytes` argument in `main()` (e.g., `50 * 1024 * 1024` for ~50 MB).

# Quick API notes

* Core function:
  `concatenate_files_by_size(file_paths, output_file_base, max_chunk_bytes=...) -> List[str]`
  Returns the list of chunk file **paths** created.
* Helpers: `bytes_human` exists for logging but isn’t used in the main flow.

