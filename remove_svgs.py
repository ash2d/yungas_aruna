# go through a folder and remove all svg files (created by matplotlib) to save space. These are not needed for the paper (pdf is good)
import os
import argparse

def remove_svg_files(folder_path, remove_non_numeric_prefix=False):
    """Remove files from the specified folder.

    By default this removes all files ending with the .svg extension.
    If remove_non_numeric_prefix is True, it will also remove any file
    whose filename does NOT start with three numeric characters (e.g. '001').
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Always consider .svg files for removal
        remove = filename.endswith('.svg')

        # Optionally remove files that don't start with three digits
        if remove_non_numeric_prefix:
            # filename may include extensions; check the base name
            base = filename
            # Check first three characters are digits
            if len(base) < 3 or not (base[0:3].isdigit()):
                remove = True

        if remove:
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove .svg files and optionally files without a 3-digit prefix from a specified folder.")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing files to remove.")
    parser.add_argument("--remove_non_numeric_prefix", action='store_true', help="Also remove any file that does not start with three digits.")
    args = parser.parse_args()
    
    if not args.folder_path:
        parser.error('--folder_path is required')

    remove_svg_files(args.folder_path, remove_non_numeric_prefix=args.remove_non_numeric_prefix)