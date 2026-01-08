import h5py
import argparse


def print_h5_structure(item, indent=0):
    """
    Recursively print the structure of an HDF5 file with indentation to show hierarchy.

    Args:
        item: HDF5 file, group, or dataset
        indent: Current indentation level
    """
    # Create indentation string
    indent_str = '    ' * indent
    
    if isinstance(item, h5py.File):
        print(f"{indent_str}File: {item.filename}")
    elif isinstance(item, h5py.Group):
        print(f"{indent_str}Group: {item.name}")
    elif isinstance(item, h5py.Dataset):
        print(f"{indent_str}Dataset: {item.name}")
        print(f"{indent_str}    Shape: {item.shape}")
        print(f"{indent_str}    Dtype: {item.dtype}")
        return
    
    # Recursively process all child items
    for key in item.keys():
        print(f"{indent_str}  Key: {key}")
        child = item[key]
        print_h5_structure(child, indent + 1)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Print the structure of an HDF5 file')
    argparser.add_argument('--file_path', type=str, required=True,
                        help='Path to the HDF5 file')
    args = argparser.parse_args()
    file_path = args.file_path

    # Open HDF5 file and print structure
    with h5py.File(file_path, 'r') as f:
        print_h5_structure(f)