import json
import os
import platform
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple

import cpuinfo
import psutil
import torch

from xdeploy.common.logger_utils import logger

__all__ = ["get_and_print_system_info", "query_gpu_info_single"]


def print_system_info_table(info: Dict[str, Any] = None) -> None:
    """
    Print system information in table format

    Args:
        info: System information dictionary, if None will be fetched automatically
    """
    if info is None:
        info = get_system_info()

    print("=" * 80)
    print("System Environment Information (for Model Acceleration Analysis)")
    print(
        "!!! The data sources may not be completely accurate and are for reference only."
    )
    print("=" * 80)

    # Print by category
    categories = {
        "GPU Information": [
            "GPU Model",
            "GPU Memory (GB)",
            "Compute Capability",
            "GPU Count",
            "Memory Bandwidth (GB/s)",
            "Memory Bus type",
            "Memory Bus width (bit)",
            "Microarchitecture",
            "Processing power (TFLOPS) Double precision (FMA)",
            "Processing power (TFLOPS) Half precision Tensor Core FP32 Accumulate",
            "Processing power (TFLOPS) Single precision (MAD or FMA)",
        ],
        "CPU Information": [
            "CPU Model",
            "CPU Cores",
            "CPU Threads",
            "CPU Frequency (GHz)",
        ],
        "Software Versions": [
            "CUDA Version",
            "PyTorch Version",
            "TorchVision Version",
            "Triton Version",
        ],
        "System Information": ["OS", "Python Version", "Architecture"],
    }

    for category, keys in categories.items():
        print(f"\n{category}:")
        print("-" * 40)
        max_key_len = max(len(key) for key in keys)
        for key in keys:
            if key in info:
                value = info[key]
                print(f"{key:<{max_key_len}} : {value}")

    print("\n" + "=" * 80)


def get_and_print_system_info() -> Dict[str, Any]:
    """
    Get and print system information, returning the information dictionary

    Returns:
        Dict[str, Any]: System information dictionary
    """
    info = get_system_info()
    print_system_info_table(info)
    return info


def get_system_info() -> Dict[str, Any]:
    """
    Get detailed information about the current runtime environment for model acceleration analysis

    Returns:
        Dict[str, Any]: Dictionary containing system information
    """

    info = {}

    def _extract_precision_metric(
        data: Dict[str, Any],
        canonical_key: str,
        required_terms: Tuple[str, ...],
    ):
        """Return preferred TFLOPS metric even if key naming varies."""

        def _valid(val: Any) -> bool:
            return val not in (None, "", "nan")

        # Prefer the canonical key when it exists
        if canonical_key in data and _valid(data[canonical_key]):
            return data[canonical_key]

        # Fall back to fuzzy key lookup (keys usually contain TFLOPS + precision)
        for k, v in data.items():
            if not isinstance(k, str) or not _valid(v):
                continue
            key_lower = k.lower()
            if "tflops" not in key_lower:
                continue
            if all(term in key_lower for term in required_terms):
                return v
        return None

    # GPU information
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device)
            total_memory = (
                torch.cuda.get_device_properties(device).total_memory
                / 1024 ** 3
            )  # GB
            compute_capability = torch.cuda.get_device_capability(device)
            compute_capability_str = (
                f"{compute_capability[0]}.{compute_capability[1]}"
            )

            info.update(
                {
                    "GPU Model": device_name,
                    "GPU Memory (GB)": round(total_memory, 2),
                    "Compute Capability": compute_capability_str,
                    "GPU Count": torch.cuda.device_count(),
                }
            )

            try:
                # Try to match by device name first
                gpu_data = query_gpu_info_single(device_name)

                if not gpu_data:
                    # If no match, prompt user for GPU model input
                    print(
                        f"\nCould not find detailed info for GPU: {device_name}"
                    )
                    user_input = input(
                        "Please enter GPU model (e.g., A100, V100, H100, 4090): "
                    ).strip()
                    if user_input:
                        gpu_data = query_gpu_info_single(user_input)
                    else:
                        gpu_data = {}

                if gpu_data:
                    # Extract key GPU specifications
                    gpu_specs = {}

                    # Memory information
                    memory_bandwidth = gpu_data.get("Memory Bandwidth (GB/s)")
                    if memory_bandwidth and memory_bandwidth != "nan":
                        gpu_specs["Memory Bandwidth (GB/s)"] = memory_bandwidth

                    memory_bus_type = gpu_data.get("Memory Bus type")
                    if memory_bus_type and memory_bus_type != "nan":
                        gpu_specs["Memory Bus type"] = memory_bus_type

                    memory_bus_width = gpu_data.get("Memory Bus width (bit)")
                    if memory_bus_width and memory_bus_width != "nan":
                        gpu_specs["Memory Bus width (bit)"] = memory_bus_width

                    # Architecture information
                    microarchitecture = gpu_data.get("Microarchitecture")
                    if microarchitecture and microarchitecture != "nan":
                        gpu_specs["Microarchitecture"] = microarchitecture

                    # Performance information
                    double_precision = _extract_precision_metric(
                        gpu_data,
                        "Processing power (TFLOPS) Double precision (FMA)",
                        ("double", "precision"),
                    )
                    if double_precision and double_precision != "nan":
                        gpu_specs[
                            "Processing power (TFLOPS) Double precision (FMA)"
                        ] = double_precision

                    half_precision_tensor = _extract_precision_metric(
                        gpu_data,
                        "Processing power (TFLOPS) Half precision Tensor Core FP32 Accumulate",
                        ("half", "precision"),
                    )
                    if (
                        half_precision_tensor
                        and half_precision_tensor != "nan"
                    ):
                        gpu_specs[
                            "Processing power (TFLOPS) Half precision Tensor Core FP32 Accumulate"
                        ] = half_precision_tensor

                    single_precision = _extract_precision_metric(
                        gpu_data,
                        "Processing power (TFLOPS) Single precision (MAD or FMA)",
                        ("single", "precision"),
                    )
                    if single_precision and single_precision != "nan":
                        gpu_specs[
                            "Processing power (TFLOPS) Single precision (MAD or FMA)"
                        ] = single_precision

                    # Update info with detailed GPU specs
                    info.update(gpu_specs)
                    logger.info(
                        f"Successfully integrated detailed GPU specs for {device_name}"
                    )

                else:
                    logger.info(
                        f"No detailed GPU information found for {device_name}"
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to retrieve detailed GPU information: {e}"
                )

        except Exception as e:
            info.update(
                {
                    "GPU Model": "N/A",
                    "GPU Memory (GB)": "N/A",
                    "Compute Capability": "N/A",
                    "GPU Count": "N/A",
                }
            )
    else:
        info.update(
            {
                "GPU Model": "No GPU available",
                "GPU Memory (GB)": "N/A",
                "Compute Capability": "N/A",
                "GPU Count": 0,
            }
        )

    # CPU information
    try:
        cpu_info = cpuinfo.get_cpu_info()
        cpu_model = cpu_info.get("brand_raw", cpu_info.get("brand", "Unknown"))
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            cpu_freq_ghz = round(cpu_freq.current / 1000, 2)
        else:
            cpu_freq_ghz = "N/A"

        info.update(
            {
                "CPU Model": cpu_model,
                "CPU Cores": psutil.cpu_count(logical=False),
                "CPU Threads": psutil.cpu_count(logical=True),
                "CPU Frequency (GHz)": cpu_freq_ghz,
            }
        )
    except Exception as e:
        info.update(
            {
                "CPU Model": "N/A",
                "CPU Cores": "N/A",
                "CPU Threads": "N/A",
                "CPU Frequency (GHz)": "N/A",
            }
        )

    # CUDA version
    try:
        cuda_version = torch.version.cuda
        if cuda_version is None:
            # Try to get from nvidia-smi
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    cuda_version = "Available (via nvidia-smi)"
                else:
                    cuda_version = "Not available"
            except:
                cuda_version = "Not available"
        info["CUDA Version"] = cuda_version
    except Exception as e:
        info["CUDA Version"] = "N/A"

    # PyTorch version
    try:
        info["PyTorch Version"] = torch.__version__
    except:
        info["PyTorch Version"] = "N/A"

    # TorchVision version
    try:
        import torchvision

        info["TorchVision Version"] = torchvision.__version__
    except ImportError:
        info["TorchVision Version"] = "Not installed"
    except:
        info["TorchVision Version"] = "N/A"

    # Triton version
    try:
        import triton

        info["Triton Version"] = triton.__version__
    except ImportError:
        info["Triton Version"] = "Not installed"
    except:
        info["Triton Version"] = "N/A"

    # System information
    try:
        info.update(
            {
                "OS": platform.system() + " " + platform.release(),
                "Python Version": platform.python_version(),
                "Architecture": platform.machine(),
            }
        )
    except:
        info.update(
            {"OS": "N/A", "Python Version": "N/A", "Architecture": "N/A"}
        )

    return info


def get_gpu_info(timeout: int = 30) -> Tuple[Dict[str, Any], str]:
    """
    Get GPU information dictionary from online API with local caching

    Args:
        timeout: Timeout in seconds for the download request (default: 30)

    Returns:
        tuple[Dict[str, Any], str]: Tuple of (GPU information dictionary, download URL)
    """
    # Cache directory and file path
    cache_dir = Path.home() / ".cache" / "xdeploy"
    cache_file = cache_dir / "gpu.json"

    # URL for GPU data
    gpu_data_url = "https://raw.githubusercontent.com/voidful/gpu-info-api/gpu-data/gpu.json"

    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cache directory: {cache_dir}")

    # Check if cache file exists and is recent (optional: could add time-based cache invalidation)
    if cache_file.exists():
        try:
            logger.info(f"Loading cached GPU data from {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                gpus = json.load(f)
            return gpus, gpu_data_url
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(
                f"Warning: Failed to read cache file {cache_file}: {e}"
            )
            # Fall through to download

    # Download and cache the GPU data
    try:
        logger.info(
            f"Downloading GPU data from {gpu_data_url} (timeout: {timeout}s)"
        )
        logger.info(f"Data will be cached to: {cache_file}")

        # Use wget to download with progress
        wget_cmd = [
            "wget",
            "--timeout",
            str(timeout),
            "--tries",
            "3",
            "--progress=bar:force",  # Force progress bar display
            "--output-document",
            str(cache_file),
            gpu_data_url,
        ]

        logger.info(f"Running: {' '.join(wget_cmd)}")
        # Don't capture output so progress bar can be displayed
        result = subprocess.run(wget_cmd, text=True)

        if result.returncode == 0:
            # Load and validate JSON
            with open(cache_file, "r", encoding="utf-8") as f:
                gpus = json.load(f)

            logger.info(
                f"Successfully cached GPU data with {len(gpus)} entries"
            )
            return gpus, gpu_data_url
        else:
            logger.error(f"wget failed with return code {result.returncode}")
            logger.error(f"wget stderr: {result.stderr}")
            return {}, gpu_data_url

    except Exception as e:
        logger.error(
            f"Error downloading GPU data: {e} , url: {gpu_data_url}, please download manually and save to {cache_file}"
        )
        # Return empty dict and URL if download fails
        return {}, gpu_data_url


def query_gpu_info(query: str) -> Dict[str, Any]:
    """
    Query GPU information by name with fuzzy matching support

    Args:
        query: GPU name to search for (case-insensitive, supports partial matches)

    Returns:
        Dict[str, Any]: Dictionary of matching GPU entries, keyed by GPU ID
    """
    # Get GPU data
    gpu_data, _ = get_gpu_info()

    if not gpu_data:
        logger.warning("No GPU data available")
        return {}

    # Normalize query: lowercase, remove extra spaces, common separators
    normalized_query = query.lower().strip()
    # Remove common separators and normalize
    normalized_query = (
        normalized_query.replace("-", "").replace("_", "").replace(" ", "")
    )

    matches = {}
    # Split by multiple separators and also keep the full normalized query
    query_terms = set()
    # Add individual terms
    for term in normalized_query.split():
        query_terms.add(term)
    # Also add the full normalized query for exact matching
    query_terms.add(normalized_query)

    for gpu_id, gpu_info in gpu_data.items():
        # Check Model name and Model fields
        search_fields = []
        if "Model name" in gpu_info and gpu_info["Model name"] != "nan":
            search_fields.append(str(gpu_info["Model name"]))
        if "Model" in gpu_info and gpu_info["Model"] != "nan":
            search_fields.append(str(gpu_info["Model"]))

        if not search_fields:
            continue

        # Normalize search text
        search_text = " ".join(search_fields).lower()
        search_text_normalized = (
            search_text.replace("-", "").replace("_", "").replace(" ", "")
        )

        # Check for matches
        match_score = 0
        matched_terms = set()

        # Priority scoring based on GPU type
        is_professional_gpu = any(
            keyword in search_text_normalized
            for keyword in [
                "a100",
                "v100",
                "h100",
                "l40",
                "a6000",
                "rtxa",
                "quadro",
                "tesla",
            ]
        )
        priority_bonus = 10 if is_professional_gpu else 0

        # Exact match gets highest score
        if normalized_query == search_text_normalized:
            match_score = 100 + priority_bonus
        # Query contained in model name
        elif normalized_query in search_text_normalized:
            match_score = 80 + priority_bonus
        # Check for GPU model patterns (more flexible than keyword list)
        # Look for common GPU naming patterns like RTX 4090, GTX 1080, A100, V100, etc.
        import re

        # More specific GPU model patterns to avoid over-matching (work on normalized query)
        gpu_patterns = [
            r"(?:rtx|gtx|rx|quadro|tesla|rtxa?)\d+",  # RTX4090, GTX1080, RX6800 (no space)
            r"(?:a\d+|v\d+|h\d+|l\d+|p\d+|k\d+)\d*",  # A100, V100, H100, L40, P100, K80, etc.
            r"\d{3,4}(?:ti|super|xt|xtx)?",  # 4090, 4080Ti, 3070Super, 6800XT, etc.
        ]

        has_gpu_pattern = any(
            re.search(pattern, normalized_query, re.IGNORECASE)
            for pattern in gpu_patterns
        )

        if has_gpu_pattern:
            # Extract potential GPU model components from query
            # Look for specific patterns in normalized query like "rtx3090", "a100", "4090ti", etc.
            specific_patterns = [
                r"(rtx\d+)",  # RTX3090
                r"(gtx\d+)",  # GTX1660
                r"(rx\d+)",  # RX6800
                r"(quadro\w*\d+)",  # QuadroRTX5000
                r"(tesla\w+)",  # TeslaT4
                r"(a\d+)",  # A100
                r"(v\d+)",  # V100
                r"(h\d+)",  # H100
                r"(l\d+)",  # L40
                r"(p\d+)",  # P100
                r"(k\d+)",  # K80
                r"(\d{3,4}(?:ti|super|xt|xtx)?)",  # 3090, 4090ti, 3070super, etc.
            ]

            extracted_terms = []
            for pattern in specific_patterns:
                pattern_matches = re.findall(
                    pattern, normalized_query, re.IGNORECASE
                )
                extracted_terms.extend(pattern_matches)

            # Remove duplicates and filter out meaningless numbers
            extracted_terms = list(set(extracted_terms))
            extracted_terms = [
                term.strip().lower().replace(" ", "")
                for term in extracted_terms
            ]

            # Filter out generic numbers that are unlikely to be specific GPU models
            # Keep GPU model numbers (typically 3-4 digits starting with 1, 2, 3, 4, etc.)
            filtered_terms = []
            for term in extracted_terms:
                # Keep GPU brand + number combinations (rtx3090, gtx1660, etc.)
                if any(
                    term.startswith(prefix)
                    for prefix in [
                        "rtx",
                        "gtx",
                        "rx",
                        "quadro",
                        "tesla",
                        "a",
                        "v",
                        "h",
                        "l",
                        "p",
                        "k",
                    ]
                ):
                    filtered_terms.append(term)
                # For pure numbers, only keep those that look like GPU model numbers
                elif term.isdigit() and len(term) >= 3:
                    num = int(term)
                    # Keep numbers that are typical GPU model numbers (1000-4999 range, or specific known models)
                    if (num >= 1000 and num <= 4999) or num in [
                        6800,
                        6700,
                        6600,
                        6500,
                        6400,
                        6300,
                        6200,
                        6100,
                    ]:
                        filtered_terms.append(term)

            extracted_terms = filtered_terms

            # Check if any extracted terms appear in GPU data
            found_match = False
            for term in extracted_terms:
                if term and len(term) >= 3:  # Only check meaningful terms
                    if term in search_text_normalized:
                        match_score = 75 + priority_bonus
                        found_match = True
                        break

            # If no exact match but we have GPU pattern, check for partial matches
            if not found_match and extracted_terms:
                for term in extracted_terms:
                    if term and len(term) >= 3:
                        # Check for partial matches (e.g., "3090" in "RTX 3090 Ti")
                        if any(
                            term_part in search_text_normalized
                            for term_part in term.split()
                        ):
                            match_score = 65 + priority_bonus
                            found_match = True
                            break
        # All query terms found in model name
        elif all(term in search_text_normalized for term in query_terms):
            match_score = 60 + priority_bonus
        # Partial matches for individual terms
        else:
            for term in query_terms:
                if term in search_text_normalized:
                    matched_terms.add(term)
            if matched_terms:
                match_score = len(matched_terms) * 20 + priority_bonus

        # Also check for common GPU naming patterns
        if match_score == 0:
            # Handle cases like "4090d" matching "GeForce RTX 4090 D"
            # or "a100" matching "A100-SXM4"
            for field in search_fields:
                field_normalized = (
                    field.lower()
                    .replace("-", "")
                    .replace("_", "")
                    .replace(" ", "")
                )
                # Check if query is a significant substring
                if (
                    len(normalized_query) >= 3
                    and normalized_query in field_normalized
                ):
                    match_score = 40 + priority_bonus
                    break
                # Check for model numbers (like "4090" in "RTX 4090")
                if any(
                    term.isdigit() and term in field_normalized
                    for term in query_terms
                ):
                    match_score = 30 + priority_bonus
                    break

        if match_score > 0:
            matches[gpu_id] = {
                "gpu_info": gpu_info,
                "match_score": match_score,
                "matched_fields": search_fields,
            }

    # Sort by match score (highest first) and return top matches
    if matches:
        sorted_matches = dict(
            sorted(
                matches.items(),
                key=lambda x: x[1]["match_score"],
                reverse=True,
            )
        )
        logger.info(
            f"Found {len(sorted_matches)} GPU matches for query '{query}'"
        )

        # Return GPU info dictionary with match scores
        result = {}
        for gpu_id, match_data in sorted_matches.items():
            result[gpu_id] = {
                **match_data["gpu_info"],
                "_match_score": match_data["match_score"],
                "_matched_fields": match_data["matched_fields"],
            }
        return result
    else:
        logger.info(f"No GPU matches found for query '{query}'")
        return {}


def select_gpu_interactive(
    query: str, max_display: int = 10
) -> Dict[str, Any]:
    """
    Interactively select a GPU from query results when multiple matches are found

    Args:
        query: GPU name to search for
        max_display: Maximum number of results to display before prompting

    Returns:
        Dict[str, Any]: Single selected GPU information, or empty dict if none selected
    """
    results = query_gpu_info(query)

    if not results:
        print(f"No GPU matches found for '{query}'")
        return {}

    if len(results) == 1:
        # Only one match, return it directly
        gpu_id = next(iter(results.keys()))
        gpu_data = results[gpu_id]
        model = gpu_data.get("Model", gpu_data.get("Model name", "Unknown"))
        print(f"Found exactly one match: {model}")
        return gpu_data

    # Multiple matches - show interactive selection
    print(f"\nFound {len(results)} GPU matches for '{query}':")
    print("=" * 80)

    # Display matches
    result_items = list(results.items())
    display_count = min(len(result_items), max_display)

    for i in range(display_count):
        gpu_id, gpu_data = result_items[i]
        model_name = gpu_data.get("Model name", "N/A")
        model = gpu_data.get("Model", "N/A")
        score = gpu_data.get("_match_score", 0)

        # Determine GPU type
        gpu_type = (
            "Professional"
            if any(
                keyword in str(model).lower()
                or keyword in str(model_name).lower()
                for keyword in [
                    "a100",
                    "v100",
                    "h100",
                    "quadro",
                    "tesla",
                    "rtxa",
                ]
            )
            else "Consumer"
        )

        print(f"{i + 1}. [{gpu_type}] {model}")
        if model_name != "N/A":
            print(f"   Name: {model_name}")
        print(f"   Score: {score}")
        print()

    if len(results) > max_display:
        print(f"... and {len(results) - max_display} more results")
        print()

    # Interactive selection
    while True:
        try:
            if len(results) > max_display:
                choice = (
                    input(
                        f"Enter number (1-{display_count}) or 'a' to show all, 'q' to quit: "
                    )
                    .strip()
                    .lower()
                )
                if choice == "q":
                    print("Selection cancelled.")
                    return {}
                elif choice == "a":
                    # Show all results
                    print(f"\nAll {len(results)} matches:")
                    print("=" * 80)
                    for i, (gpu_id, gpu_data) in enumerate(result_items):
                        model = gpu_data.get(
                            "Model", gpu_data.get("Model name", "Unknown")
                        )
                        print(f"{i + 1}. {model}")
                    print()
                    choice = input(
                        f"Enter number (1-{len(results)}): "
                    ).strip()
                else:
                    choice = choice
            else:
                choice = (
                    input(f"Enter number (1-{len(results)}), or 'q' to quit: ")
                    .strip()
                    .lower()
                )
                if choice == "q":
                    print("Selection cancelled.")
                    return {}

            choice_num = int(choice)
            if 1 <= choice_num <= len(result_items):
                selected_gpu_id, selected_gpu_data = result_items[
                    choice_num - 1
                ]
                model = selected_gpu_data.get(
                    "Model", selected_gpu_data.get("Model name", "Unknown")
                )
                print(f"\nSelected: {model}")
                return selected_gpu_data
            else:
                print(
                    f"Invalid choice. Please enter a number between 1 and {len(result_items)}."
                )

        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nSelection cancelled.")
            return {}


def query_gpu_info_single(query: str) -> Dict[str, Any]:

    """
    Query GPU information and return a single selected GPU (interactive if multiple matches)

    Args:
        query: GPU name to search for

    Returns:
        Dict[str, Any]: Single GPU information dictionary
    """
    from pprint import pprint

    data = select_gpu_interactive(query)
    pprint(data, indent=2, width=80, sort_dicts=True)
    return data
