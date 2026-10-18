"""Pinned model artifacts verified in the Python 3.14 CPU experiment."""
import hashlib
from pathlib import Path

MODELS = {'gigaam': {'repo': 'istupakov/gigaam-v3-onnx',
            'revision': '322c3b29492673eb7d0b434bfa9dfb8653e34d02',
            'files': {'config.json': '0641fcf73f4af791c73f05083e38a658ff5dcbee3534a0c61396c10f4b97f0fe',
                      'v3_e2e_rnnt_decoder.onnx': '7b0a16d67fd2cb37061decc93c69e364a9ab27afee3c57495d55b1c974cf7231',
                      'v3_e2e_rnnt_encoder.onnx': 'cd60b3764a832e8560ae6d3ad0b10adc1a42ffae412b9476f25620aae4f4a508',
                      'v3_e2e_rnnt_joint.onnx': '602ff7017a93311aad34df1437c8d7f49911353c13d6eae7a6ee7b041339465c',
                      'v3_e2e_rnnt_vocab.txt': '39abae20e692998290c574e606f11a9edef2902a1995463fcff63d1490cf22b7'}},
 'silero': {'repo': 'istupakov/silero-vad-onnx',
            'revision': 'b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6',
            'files': {'config.json': '1094039d370c82889582ba739a3d1caac5754c8b3a17a66a534200c9f72086e2',
                      'silero_vad.onnx': '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'}}}

def verify_models(root):
    for name, spec in MODELS.items():
        for filename, expected in spec["files"].items():
            path = Path(root) / name / filename
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError("Модели не подготовлены или повреждены. Выполните python -m personal_assistent.setup_voice")

def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
