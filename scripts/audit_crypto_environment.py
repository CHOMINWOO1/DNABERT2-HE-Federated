from __future__ import annotations

"""Read-only provenance audit for the installed TenSEAL/SEAL environment.

The script reads the existing virtual environment, validates the installed
wheel payload against dist-info/RECORD, inspects native PE imports, and uses
the public TenSEAL/SEAL Python API to validate the paper's CKKS parameters.
It does not install, download, or modify packages. The only write is the JSON
path explicitly supplied with --output.
"""

import argparse
import base64
import csv
import hashlib
import json
import platform
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone
from email.parser import Parser
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "crypto-environment-audit-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "paper_extension_crypto_environment_audit.json"
CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MOD_BIT_SIZES = [60, 40, 60]
CKKS_SCALE_BITS = 40


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(raw.encode("utf-8"))


def urlsafe_record_hash(hex_digest: str) -> str:
    raw = bytes.fromhex(hex_digest)
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def project_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_key_value_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "path": project_relative(path), "values": {}}
    message = Parser().parsestr(path.read_text(encoding="utf-8"))
    values: dict[str, Any] = {}
    for key in message.keys():
        items = message.get_all(key, [])
        values[key] = items[0] if len(items) == 1 else items
    return {
        "present": True,
        "path": project_relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "values": values,
    }


def audit_pip_wheel_cache() -> dict[str, Any]:
    command = [sys.executable, "-m", "pip", "cache", "list", "tenseal", "--format=abspath"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    paths = [Path(line.strip()) for line in completed.stdout.splitlines() if line.strip().lower().endswith(".whl")]
    archives = [
        {
            "path": project_relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
        if path.is_file()
    ]
    return {
        "command": command,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "archive_count": len(archives),
        "archives": archives,
        "archive_sha256_status": (
            "available in archives list"
            if archives
            else "not recoverable from the installed tree because pip cache contained no TenSEAL wheel archive"
        ),
    }


def read_record(dist_info: Path) -> tuple[list[dict[str, str | None]], dict[str, dict[str, str | None]]]:
    record_path = dist_info / "RECORD"
    rows: list[dict[str, str | None]] = []
    by_path: dict[str, dict[str, str | None]] = {}
    with record_path.open("r", encoding="utf-8", newline="") as handle:
        for name, encoded_hash, encoded_size in csv.reader(handle):
            row = {
                "path": name.replace("\\", "/"),
                "record_hash": encoded_hash or None,
                "record_bytes": encoded_size or None,
            }
            rows.append(row)
            by_path[str(row["path"])] = row
    return rows, by_path


def distribution_files(site_packages: Path, package_dir: Path, dist_info: Path, record_rows: Iterable[dict[str, Any]]) -> list[Path]:
    files: set[Path] = set()
    files.update(path for path in package_dir.rglob("*") if path.is_file())
    files.update(path for path in dist_info.rglob("*") if path.is_file())
    for row in record_rows:
        path = site_packages / str(row["path"])
        if path.is_file():
            files.add(path)
    for pattern in ("_tenseal_cpp*.pyd", "_sealapi_cpp*.pyd", "*tenseal*.dll", "*sealapi*.dll"):
        files.update(path for path in site_packages.glob(pattern) if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(site_packages).as_posix())


def audit_installed_files(site_packages: Path, package_dir: Path, dist_info: Path) -> dict[str, Any]:
    record_rows, record_by_path = read_record(dist_info)
    actual_paths = distribution_files(site_packages, package_dir, dist_info, record_rows)
    actual_names = {path.relative_to(site_packages).as_posix() for path in actual_paths}
    records: list[dict[str, Any]] = []
    for path in actual_paths:
        name = path.relative_to(site_packages).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        wheel_row = record_by_path.get(name)
        if wheel_row is None:
            status = "not_listed_in_record"
        elif wheel_row["record_hash"] is None:
            status = "record_unhashed"
        else:
            algorithm, expected = str(wheel_row["record_hash"]).split("=", 1)
            if algorithm != "sha256":
                status = f"unsupported_record_hash:{algorithm}"
            elif expected != urlsafe_record_hash(digest):
                status = "record_hash_mismatch"
            elif wheel_row["record_bytes"] is not None and int(str(wheel_row["record_bytes"])) != size:
                status = "record_size_mismatch"
            else:
                status = "match"
        records.append(
            {
                "path": name,
                "bytes": size,
                "sha256": digest,
                "record_hash": None if wheel_row is None else wheel_row["record_hash"],
                "record_bytes": None if wheel_row is None else wheel_row["record_bytes"],
                "record_status": status,
                "runtime_generated": "__pycache__/" in name or name.endswith(".pyc"),
            }
        )

    missing_from_disk = sorted(str(row["path"]) for row in record_rows if str(row["path"]) not in actual_names)
    payload_records = [
        {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in records
        if not row["runtime_generated"] and not row["path"].endswith(".dist-info/RECORD")
    ]
    current_records = [
        {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in records
    ]
    mismatch_rows = [row for row in records if row["record_status"] not in {"match", "record_unhashed"}]
    return {
        "record": {
            "path": project_relative(dist_info / "RECORD"),
            "bytes": (dist_info / "RECORD").stat().st_size,
            "sha256": sha256_file(dist_info / "RECORD"),
            "row_count": len(record_rows),
            "missing_from_disk": missing_from_disk,
            "mismatch_count": len(mismatch_rows),
            "mismatches": mismatch_rows,
        },
        "file_count": len(records),
        "wheel_payload_file_count": len(payload_records),
        "runtime_generated_file_count": sum(bool(row["runtime_generated"]) for row in records),
        "wheel_payload_aggregate_sha256": json_sha256(payload_records),
        "current_installed_tree_aggregate_sha256": json_sha256(current_records),
        "files": records,
    }


def _rva_to_offset(rva: int, sections: list[dict[str, int]]) -> int:
    for section in sections:
        start = section["virtual_address"]
        span = max(section["virtual_size"], section["raw_size"])
        if start <= rva < start + span:
            return section["raw_pointer"] + (rva - start)
    raise ValueError(f"RVA 0x{rva:x} is not mapped by a PE section")


def _read_c_string(data: bytes, offset: int, max_length: int = 1024) -> str:
    if offset < 0 or offset >= len(data):
        raise ValueError(f"string offset out of range: {offset}")
    end = data.find(b"\0", offset, min(len(data), offset + max_length))
    if end < 0:
        end = min(len(data), offset + max_length)
    return data[offset:end].decode("ascii", errors="replace")


def parse_pe_imports(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise ValueError("not an MZ executable")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise ValueError("missing PE signature")
    coff = pe_offset + 4
    machine, section_count, pe_timestamp, _, _, optional_size, characteristics = struct.unpack_from(
        "<HHIIIHH", data, coff
    )
    optional = coff + 20
    magic = struct.unpack_from("<H", data, optional)[0]
    if magic == 0x20B:
        pe_kind = "PE32+"
        data_directory = optional + 112
        image_base = struct.unpack_from("<Q", data, optional + 24)[0]
    elif magic == 0x10B:
        pe_kind = "PE32"
        data_directory = optional + 96
        image_base = struct.unpack_from("<I", data, optional + 28)[0]
    else:
        raise ValueError(f"unsupported PE optional-header magic 0x{magic:x}")
    section_offset = optional + optional_size
    sections: list[dict[str, int]] = []
    for index in range(section_count):
        offset = section_offset + index * 40
        virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from("<IIII", data, offset + 8)
        sections.append(
            {
                "virtual_size": virtual_size,
                "virtual_address": virtual_address,
                "raw_size": raw_size,
                "raw_pointer": raw_pointer,
            }
        )

    def directory(index: int) -> tuple[int, int]:
        return struct.unpack_from("<II", data, data_directory + index * 8)

    imports: list[str] = []
    import_rva, import_size = directory(1)
    if import_rva and import_size:
        offset = _rva_to_offset(import_rva, sections)
        for _ in range(4096):
            fields = struct.unpack_from("<IIIII", data, offset)
            if not any(fields):
                break
            imports.append(_read_c_string(data, _rva_to_offset(fields[3], sections)))
            offset += 20

    delay_imports: list[str] = []
    delay_rva, delay_size = directory(13)
    if delay_rva and delay_size:
        offset = _rva_to_offset(delay_rva, sections)
        for _ in range(4096):
            fields = struct.unpack_from("<IIIIIIII", data, offset)
            if not any(fields):
                break
            attributes, name_value = fields[0], fields[1]
            name_rva = name_value if attributes & 1 else name_value - image_base
            delay_imports.append(_read_c_string(data, _rva_to_offset(name_rva, sections)))
            offset += 32

    return {
        "pe_kind": pe_kind,
        "machine_hex": f"0x{machine:04x}",
        "section_count": section_count,
        "pe_timestamp_unix": pe_timestamp,
        "characteristics_hex": f"0x{characteristics:04x}",
        "imports": sorted(set(imports), key=str.lower),
        "delay_imports": sorted(set(delay_imports), key=str.lower),
    }


def printable_string_evidence(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    strings = [match.group().decode("ascii", errors="replace") for match in re.finditer(rb"[ -~]{4,}", data)]
    semver_candidates = sorted(
        {
            value.strip()
            for value in strings
            if len(value) <= 180 and re.search(r"\b\d+\.\d+\.\d+\b", value)
        }
    )
    return {
        "contains_seal_cpp_symbol_or_rtti": b"@seal@@" in data or b"SEALContext@seal" in data,
        "contains_literal_microsoft_seal": b"Microsoft SEAL" in data,
        "contains_literal_seal_version": b"SEAL_VERSION" in data or b"SEAL version" in data,
        "semver_candidates": semver_candidates,
        "unambiguous_microsoft_seal_patch_version_found": False,
        "note": (
            "Printable strings expose SEAL C++ symbols but no unambiguous Microsoft SEAL patch version. "
            "Observed semver strings belong to bundled compression/protobuf dependencies and are not used as SEAL evidence."
        ),
    }


def audit_native_binaries(site_packages: Path, installed_files: dict[str, Any]) -> dict[str, Any]:
    binary_rows = [
        row for row in installed_files["files"] if Path(str(row["path"])).suffix.lower() in {".pyd", ".dll"}
    ]
    output: list[dict[str, Any]] = []
    all_imports: set[str] = set()
    for row in binary_rows:
        path = site_packages / str(row["path"])
        pe = parse_pe_imports(path)
        evidence = printable_string_evidence(path)
        all_imports.update(pe["imports"])
        all_imports.update(pe["delay_imports"])
        output.append({**row, "pe": pe, "printable_string_evidence": evidence})
    seal_imports = sorted(
        name for name in all_imports if re.search(r"(?i)(?:^|[_-])seal(?:[._-]|$)", name)
    )
    adjacent_seal_dlls = sorted(
        path.name
        for path in site_packages.glob("*.dll")
        if re.search(r"(?i)(?:tenseal|sealapi|(?:^|[_-])seal(?:[._-]|$))", path.name)
    )
    return {
        "binary_count": len(output),
        "binaries": output,
        "all_imported_libraries": sorted(all_imports, key=str.lower),
        "external_seal_imports": seal_imports,
        "adjacent_seal_named_dlls": adjacent_seal_dlls,
        "linkage_conclusion": (
            "Microsoft SEAL code is bundled/static-linked into the extension modules: the PE import tables contain "
            "no SEAL-named DLL, no adjacent SEAL DLL is present, and SEAL C++ symbols/RTTI occur in the binaries."
            if output and not seal_imports and not adjacent_seal_dlls
            else "Static linkage could not be concluded from the available PE evidence."
        ),
    }


def validate_seal_context(sa: Any, parms: Any, level: Any) -> dict[str, Any]:
    context = sa.SEALContext(parms, True, level)
    qualifiers = context.key_context_data().qualifiers()
    return {
        "level": str(level),
        "parameters_set": bool(context.parameters_set()),
        "parameters_error_name": str(context.parameters_error_name()),
        "parameters_error_message": str(context.parameters_error_message()),
        "qualifier_parameters_set": bool(qualifiers.parameters_set()),
        "qualifier_sec_level": str(qualifiers.sec_level),
    }


def dynamic_api_audit() -> dict[str, Any]:
    sys.dont_write_bytecode = True
    import tenseal as ts
    import tenseal.sealapi as sa
    import tenseal.sealapi.util as seal_util

    header = sa.Serialization.SEALHeader()
    parms = sa.EncryptionParameters(sa.SCHEME_TYPE.CKKS)
    parms.set_poly_modulus_degree(CKKS_POLY_MODULUS_DEGREE)
    moduli = sa.CoeffModulus.Create(CKKS_POLY_MODULUS_DEGREE, CKKS_COEFF_MOD_BIT_SIZES)
    parms.set_coeff_modulus(moduli)
    actual_bits = [int(modulus.bit_count()) for modulus in moduli]
    actual_values = [int(modulus.value()) for modulus in moduli]
    total_bits = sum(actual_bits)
    validations = {
        "tc128": validate_seal_context(sa, parms, sa.SEC_LEVEL_TYPE.TC128),
        "tc192": validate_seal_context(sa, parms, sa.SEC_LEVEL_TYPE.TC192),
        "tc256": validate_seal_context(sa, parms, sa.SEC_LEVEL_TYPE.TC256),
        "none": validate_seal_context(sa, parms, sa.SEC_LEVEL_TYPE.NONE),
    }
    tc_limits = {
        "128": int(sa.CoeffModulus.MaxBitCount(CKKS_POLY_MODULUS_DEGREE, sa.SEC_LEVEL_TYPE.TC128)),
        "192": int(sa.CoeffModulus.MaxBitCount(CKKS_POLY_MODULUS_DEGREE, sa.SEC_LEVEL_TYPE.TC192)),
        "256": int(sa.CoeffModulus.MaxBitCount(CKKS_POLY_MODULUS_DEGREE, sa.SEC_LEVEL_TYPE.TC256)),
    }
    tq_limits = {
        "128": int(seal_util.seal_he_std_parms_128_tq(CKKS_POLY_MODULUS_DEGREE)),
        "192": int(seal_util.seal_he_std_parms_192_tq(CKKS_POLY_MODULUS_DEGREE)),
        "256": int(seal_util.seal_he_std_parms_256_tq(CKKS_POLY_MODULUS_DEGREE)),
    }

    secret_context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=CKKS_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=CKKS_COEFF_MOD_BIT_SIZES,
    )
    secret_context.global_scale = 2**CKKS_SCALE_BITS
    public_blob = secret_context.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=False,
        save_relin_keys=False,
    )
    secret_blob_size = len(secret_context.serialize(save_secret_key=True))
    public_context = ts.context_from(public_blob)
    sample = [0.125, -1.25, 3.5]
    encrypted_a = ts.ckks_vector(public_context, sample)
    encrypted_b = ts.ckks_vector(public_context, sample)
    serialized_a = encrypted_a.serialize()
    serialized_b = encrypted_b.serialize()
    summed = encrypted_a + encrypted_b
    recovered = ts.ckks_vector_from(secret_context, summed.serialize()).decrypt()
    expected = [2.0 * value for value in sample]
    max_abs_error = max(abs(float(left) - float(right)) for left, right in zip(recovered, expected))

    return {
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
            "compiler": platform.python_compiler(),
            "executable": project_relative(Path(sys.executable)),
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
        },
        "tenseal": {
            "version": str(ts.__version__),
            "module_path": project_relative(Path(ts.__file__)),
        },
        "microsoft_seal": {
            "serialization_header": {
                "magic": int(header.magic),
                "header_size": int(header.header_size),
                "version_major": int(header.version_major),
                "version_minor": int(header.version_minor),
                "compression_mode": str(header.compr_mode),
            },
            "linked_version_conclusion": f"{int(header.version_major)}.{int(header.version_minor)}.x",
            "patch_version_status": (
                "not exposed by TenSEAL metadata, the public Serialization.SEALHeader API, or unambiguous native strings"
            ),
        },
        "ckks_parameter_validation": {
            "poly_modulus_degree": CKKS_POLY_MODULUS_DEGREE,
            "requested_coeff_mod_bit_sizes": CKKS_COEFF_MOD_BIT_SIZES,
            "actual_coeff_mod_bit_sizes": actual_bits,
            "actual_coeff_modulus_values": actual_values,
            "total_coeff_modulus_bits": total_bits,
            "global_scale": 2**CKKS_SCALE_BITS,
            "global_scale_bits": CKKS_SCALE_BITS,
            "slots": CKKS_POLY_MODULUS_DEGREE // 2,
            "tc_max_total_coeff_modulus_bits": tc_limits,
            "tq_reference_max_total_coeff_modulus_bits": tq_limits,
            "explicit_seal_context_validation": validations,
            "tc128_margin_bits": tc_limits["128"] - total_bits,
            "tc192_excess_bits": total_bits - tc_limits["192"],
            "tc256_excess_bits": total_bits - tc_limits["256"],
            "conclusion": (
                "The parameter set passes the installed SEAL TC128 validator and fails TC192 and TC256. "
                "This supports a TC128 parameter-compliance statement, not a higher-level or end-to-end security guarantee."
            ),
            "scale_note": "The 2^40 CKKS scale controls approximate-number precision; it is not the security level.",
        },
        "ephemeral_functional_check": {
            "secret_context_is_private": bool(secret_context.is_private()),
            "secret_context_has_secret_key": bool(secret_context.has_secret_key()),
            "secret_context_has_public_key": bool(secret_context.has_public_key()),
            "public_context_is_public": bool(public_context.is_public()),
            "public_context_has_secret_key": bool(public_context.has_secret_key()),
            "public_context_has_public_key": bool(public_context.has_public_key()),
            "public_context_bytes": len(public_blob),
            "secret_context_bytes": secret_blob_size,
            "same_plaintext_encryptions_have_different_serializations": serialized_a != serialized_b,
            "addition_max_abs_error": max_abs_error,
            "key_or_ciphertext_material_written_to_disk": False,
            "note": "All keys and ciphertexts in this check were ephemeral in-memory objects.",
        },
    }


def source_hits(path: Path, expressions: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    output: dict[str, list[dict[str, Any]]] = {}
    for label, expression in expressions.items():
        pattern = re.compile(expression)
        output[label] = [
            {"line": index, "text": line.strip()}
            for index, line in enumerate(lines, start=1)
            if pattern.search(line)
        ]
    return output


def runner_source_audit() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "scripts" / "run_fedhe_progress_v2.py"
    crypto_path = PROJECT_ROOT / "scripts" / "run_fedhe_experiment.py"
    full_ft_path = PROJECT_ROOT / "scripts" / "run_fedhe_main_experiment.py"
    files = [config_path, crypto_path, full_ft_path]
    return {
        "source_files": [
            {"path": project_relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
        "parameter_configuration_evidence": source_hits(
            config_path,
            {
                "poly_modulus_degree": r"ckks_poly_modulus_degree=8192",
                "coeff_mod_bit_sizes": r"ckks_coeff_mod_bits=\[60, 40, 60\]",
                "scale_bits": r"ckks_scale_bits=40",
            },
        ),
        "single_process_key_topology_evidence": source_hits(
            crypto_path,
            {
                "secret_context_created": r"self\.secret_context\s*=\s*ts\.context",
                "public_blob_excludes_secret_key": r"save_secret_key=False",
                "public_context_created": r"self\.public_context\s*=\s*ts\.context_from",
                "public_context_encrypts": r"ts\.ckks_vector\(self\.public_context",
                "secret_context_decrypts": r"ts\.ckks_vector_from\(self\.secret_context",
            },
        ),
        "full_ft_uses_same_backend": source_hits(
            full_ft_path,
            {"backend_constructor": r"self\.backend\s*=\s*pilot\.CKKSAdditiveAggregator\(config\)"},
        ),
    }


def build_audit(venv_dir: Path, output_path: Path | None) -> dict[str, Any]:
    site_packages = venv_dir / "Lib" / "site-packages"
    package_dir = site_packages / "tenseal"
    dist_candidates = sorted(site_packages.glob("tenseal-*.dist-info"))
    if len(dist_candidates) != 1:
        raise RuntimeError(f"Expected exactly one tenseal dist-info directory, found: {dist_candidates}")
    dist_info = dist_candidates[0]
    if not package_dir.is_dir():
        raise RuntimeError(f"TenSEAL package directory is missing: {package_dir}")

    installed_files = audit_installed_files(site_packages, package_dir, dist_info)
    native = audit_native_binaries(site_packages, installed_files)
    metadata = parse_key_value_file(dist_info / "METADATA")
    wheel = parse_key_value_file(dist_info / "WHEEL")
    pip_cache = audit_pip_wheel_cache()
    installer_path = dist_info / "INSTALLER"
    direct_url_path = dist_info / "direct_url.json"
    requested_path = dist_info / "REQUESTED"
    dynamic = dynamic_api_audit()
    runner = runner_source_audit()

    audit = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_mode": {
            "read_only_environment_audit": True,
            "package_install_download_or_change_performed": False,
            "h3k9_result_artifacts_accessed": False,
            "only_output_written": project_relative(output_path) if output_path is not None else None,
        },
        "environment": {
            "project_root": str(PROJECT_ROOT),
            "venv_dir": project_relative(venv_dir),
            "pyvenv_cfg": {
                "path": project_relative(venv_dir / "pyvenv.cfg"),
                "bytes": (venv_dir / "pyvenv.cfg").stat().st_size,
                "sha256": sha256_file(venv_dir / "pyvenv.cfg"),
                "text": (venv_dir / "pyvenv.cfg").read_text(encoding="utf-8").splitlines(),
            },
            "dynamic_api": dynamic,
        },
        "distribution": {
            "normalized_name": "tenseal",
            "version": dynamic["tenseal"]["version"],
            "package_dir": project_relative(package_dir),
            "dist_info_dir": project_relative(dist_info),
            "installer": {
                "path": project_relative(installer_path),
                "value": installer_path.read_text(encoding="utf-8").strip(),
                "bytes": installer_path.stat().st_size,
                "sha256": sha256_file(installer_path),
            },
            "requested_marker": {
                "present": requested_path.is_file(),
                "bytes": requested_path.stat().st_size if requested_path.is_file() else None,
                "sha256": sha256_file(requested_path) if requested_path.is_file() else None,
            },
            "direct_url": {
                "present": direct_url_path.is_file(),
                "path": project_relative(direct_url_path),
                "value": json.loads(direct_url_path.read_text(encoding="utf-8")) if direct_url_path.is_file() else None,
                "interpretation": (
                    "No direct_url.json is installed; a VCS commit or local source path cannot be recovered from PEP 610 metadata."
                    if not direct_url_path.is_file()
                    else "PEP 610 origin metadata is available."
                ),
            },
            "metadata": metadata,
            "wheel": {
                **wheel,
                "pip_cache": pip_cache,
            },
            "installed_files": installed_files,
            "native_binaries": native,
        },
        "runner_source_audit": runner,
        "security_claim": {
            "supported_korean": (
                "본 구현의 CKKS 파라미터(N=8192, coeff-modulus bit sizes=[60,40,60], 총 160 bits)는 "
                "설치된 Microsoft SEAL 4.3.x의 HomomorphicEncryption.org TC128 파라미터 검증을 통과했다."
            ),
            "supported_english": (
                "The CKKS parameter set (N=8192; coefficient-modulus bit sizes [60,40,60], 160 bits in total) "
                "passes the HomomorphicEncryption.org TC128 parameter validator in the installed Microsoft SEAL 4.3.x build."
            ),
            "claim_scope": (
                "Parameter compliance for this SEAL build. It is not an independent cryptanalysis, software audit, "
                "side-channel assessment, protocol proof, or end-to-end deployment guarantee."
            ),
            "unsupported_claims": [
                "192-bit or 256-bit security for this parameter set",
                "a fully isolated key authority",
                "threshold or multiparty CKKS",
                "protection against a coordinator that can read the training process memory",
                "malicious-client security, collusion resistance, differential privacy, or traffic-analysis resistance",
                "an exact Microsoft SEAL patch version beyond the observed 4.3 major/minor serialization version",
            ],
        },
        "key_authority_and_threat_model": {
            "classification": "single-key logical key-authority prototype in one Python process",
            "protected_surface_in_this_experiment": [
                "Client update vectors are represented as randomized CKKS ciphertexts on the simulated upload/aggregation path.",
                "The aggregation path adds ciphertexts using a public context that excludes the secret key.",
                "Only the cohort aggregate is passed to the secret context for decryption in the implemented control flow.",
            ],
            "limitations": [
                "The secret and public contexts coexist in the same Python process and address space.",
                "The same process creates the key pair, encrypts simulated client updates, aggregates ciphertexts, and decrypts the aggregate.",
                "A single secret key is used; compromise of that key holder compromises all ciphertexts under that key.",
                "There is no separate machine/process identity, HSM, remote attestation, threshold decryption, or multiparty key generation.",
                "The implementation does not model authenticated key distribution, secure transport, replay protection, client dropout, or collusion.",
                "The decrypted cohort aggregate remains visible to the key holder; small-cohort and differencing leakage are not mitigated by HE alone.",
                "No differential privacy guarantee is provided, and HE does not by itself prevent malicious or poisoned client updates.",
                "Microsoft SEAL does not claim plaintext-independent execution time; side channels are outside this experiment.",
            ],
            "paper_safe_interpretation": (
                "따라서 본 결과는 암호문 가중합의 기능성, 수치 충실도, 직렬화 비용을 평가한다. "
                "실제 기관 간 배포에서 서버로부터 비밀키를 격리했다는 증거로 해석하지 않는다."
            ),
            "deployment_requirements_for_stronger_claim": [
                "Run key generation and decryption in an independently administered KA process or service with authenticated channels and access control.",
                "Use threshold/multiparty CKKS if no single organization may hold the complete decryption key.",
                "Define minimum cohort size, dropout and collusion assumptions, key rotation, audit logging, and aggregate-release policy.",
                "Add secure aggregation and/or differential privacy when the released aggregate itself is sensitive.",
                "Conduct implementation, serialization, and side-channel review for the deployed threat model.",
            ],
        },
        "references": [
            {
                "title": "Microsoft SEAL modulus.h (security-level API and MaxBitCount)",
                "url": "https://github.com/microsoft/SEAL/blob/main/native/src/seal/modulus.h",
                "use": "SEAL TC128/TC192/TC256 definitions and coefficient-modulus limits",
            },
            {
                "title": "Microsoft SEAL BFV basics example (8192 -> 218-bit TC128 limit)",
                "url": "https://github.com/microsoft/SEAL/blob/main/native/examples/1_bfv_basics.cpp",
                "use": "Official explanation of total coefficient-modulus bit length and the HomomorphicEncryption.org limit",
            },
            {
                "title": "Homomorphic Encryption Standard v1.1",
                "url": "https://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf",
                "use": "Standard parameter tables and security-model context",
            },
            {
                "title": "Correct use of Microsoft SEAL",
                "url": "https://github.com/microsoft/SEAL/security",
                "use": "Decryption handling and timing/side-channel cautions",
            },
            {
                "title": "TenSEAL repository",
                "url": "https://github.com/OpenMined/TenSEAL",
                "use": "TenSEAL architecture and Microsoft SEAL dependency",
            },
        ],
    }
    audit["audit_payload_sha256_excluding_self_and_timestamp"] = json_sha256(
        {key: value for key, value in audit.items() if key not in {"generated_at_utc", "audit_payload_sha256_excluding_self_and_timestamp"}}
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv-dir", type=Path, default=PROJECT_ROOT / ".venv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run all checks and print the payload hash without writing the JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = None if args.verify_only else args.output.resolve()
    audit = build_audit(args.venv_dir.resolve(), output)
    if args.verify_only:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "payload_sha256": audit["audit_payload_sha256_excluding_self_and_timestamp"]}))
        return
    assert output is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(f"sha256={sha256_file(output)}")
    print(f"payload_sha256={audit['audit_payload_sha256_excluding_self_and_timestamp']}")


if __name__ == "__main__":
    main()
