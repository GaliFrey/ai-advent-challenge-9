"""Download pinned voice models. No Telegram or DeepSeek credentials needed."""
import os
import shutil
import sys
from dotenv import load_dotenv
from .voice import ROOT, VoiceConfig
from .voice_models import MODELS, file_hash, verify_models


def prepare(config):
    if not shutil.which('ffmpeg'):
        raise ValueError('Установите системный ffmpeg перед подготовкой голоса.')
    from huggingface_hub import hf_hub_download
    for name, spec in MODELS.items():
        directory = config.model_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        for filename, digest in spec['files'].items():
            target = directory / filename
            if target.is_file() and file_hash(target) == digest:
                continue
            hf_hub_download(spec['repo'], filename, revision=spec['revision'], local_dir=directory,
                            force_download=target.exists())
    verify_models(config.model_dir)


def main():
    os.umask(0o077)
    load_dotenv(ROOT / '.env')
    try:
        config = VoiceConfig.load()
        prepare(config)
    except ImportError:
        print('Установите зависимости: uv sync --locked --extra voice', file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print('Не удалось подготовить модели: ' + type(exc).__name__, file=sys.stderr)
        return 1
    print('GigaAM и Silero VAD подготовлены: ' + str(config.model_dir))
    return 0


if __name__ == '__main__':
    sys.exit(main())
