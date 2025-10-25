import pandas as pd
import os

def split_csv(file_path, chunk_size_mb=20, output_dir="data_chunks"):
    """
    Splits a large CSV file into multiple smaller CSV files.

    Args:
        file_path (str): Path to the large CSV file.
        chunk_size_mb (int): Maximum size of each output CSV file in MB.
        output_dir (str): Directory to save the smaller CSV files.
    """
    if not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)
    else:
        print(f"Directory already exists: {output_dir}")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    total_rows = len(df)
    
    # Estimate rows per chunk. This is an approximation as actual file size depends on data.
    # We'll aim for slightly less than 25MB to be safe, e.g., 20MB.
    # This calculation is a heuristic; actual file size will vary.
    # A more robust solution would involve writing chunks and checking their size.
    
    # Let's assume a rough average row size. We'll read a small sample to estimate.
    sample_df = df.head(1000)
    sample_file_path = "temp_sample.csv"
    sample_df.to_csv(sample_file_path, index=False)
    sample_size_bytes = os.path.getsize(sample_file_path)
    os.remove(sample_file_path)

    if sample_size_bytes == 0:
        print("Error: Sample file size is 0. Cannot estimate row size.")
        return

    avg_row_size_bytes = sample_size_bytes / 1000
    
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    rows_per_chunk = int(chunk_size_bytes / avg_row_size_bytes)
    
    if rows_per_chunk == 0:
        print("Error: Rows per chunk is 0. Chunk size might be too small or row size too large.")
        return

    num_chunks = (total_rows + rows_per_chunk - 1) // rows_per_chunk

    print(f"Splitting {file_path} into {num_chunks} chunks of approximately {chunk_size_mb}MB each.")
    print(f"Estimated rows per chunk: {rows_per_chunk}")

    for i in range(num_chunks):
        start_row = i * rows_per_chunk
        end_row = min((i + 1) * rows_per_chunk, total_rows)
        chunk_df = df.iloc[start_row:end_row]
        
        output_file = os.path.join(output_dir, f"Suicide_Detection_part_{i+1}.csv")
        try:
            chunk_df.to_csv(output_file, index=False)
            print(f"Created {output_file} with {len(chunk_df)} rows.")
        except Exception as e:
            print(f"Error writing chunk to {output_file}: {e}")

if __name__ == "__main__":
    original_csv_path = "Suicide_Detection.csv"
    split_csv(original_csv_path)
