from dataclasses import dataclass, field
from pathlib import Path
import os
import math
from dotenv import load_dotenv
from .voice import VoiceConfig

ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class Config:
    telegram_token: str
    owner_id: int
    deepseek_key: str
    db_path: Path
    temperature: float = 0.0
    max_tokens: int = 2000
    context_budget: int = 60000
    context_reserve: int = 2000
    processing_timeout: float = 120
    voice: VoiceConfig = field(default_factory=VoiceConfig)

    @classmethod
    def load(cls):
        load_dotenv(ROOT / '.env')
        required = ('TELEGRAM_BOT_TOKEN', 'TELEGRAM_OWNER_ID', 'DEEPSEEK_API_KEY')
        missing = [name for name in required if not os.environ.get(name, '').strip()]
        if missing:
            raise ValueError('Задайте в .env: ' + ', '.join(missing))
        try:
            owner = int(os.environ['TELEGRAM_OWNER_ID'])
            if owner <= 0:
                raise ValueError
        except ValueError:
            raise ValueError('TELEGRAM_OWNER_ID должен быть положительным числом') from None
        try:
            temperature = float(os.getenv('DEEPSEEK_TEMPERATURE', '0'))
            max_tokens = int(os.getenv('DEEPSEEK_MAX_TOKENS', '2000'))
            if not math.isfinite(temperature) or not 0 <= temperature <= 2:
                raise ValueError
            if not 128 <= max_tokens <= 8192:
                raise ValueError
        except ValueError:
            raise ValueError('DEEPSEEK_TEMPERATURE: число от 0 до 2; DEEPSEEK_MAX_TOKENS: целое от 128 до 8192') from None
        try:
            context_budget = int(os.getenv('ASSISTANT_CONTEXT_BUDGET', '60000'))
            context_reserve = int(os.getenv('ASSISTANT_CONTEXT_RESERVE', str(max(max_tokens, 1000))))
            if not max(max_tokens, 1000) <= context_reserve < context_budget <= 1000000:
                raise ValueError
        except ValueError:
            raise ValueError('Бюджет: целое до 1000000, больше резерва; резерв не меньше max_tokens и 1000') from None
        try:
            processing_timeout = float(os.getenv('ASSISTANT_PROCESSING_TIMEOUT', '120'))
            if not math.isfinite(processing_timeout) or processing_timeout <= 0:
                raise ValueError
        except ValueError:
            raise ValueError('ASSISTANT_PROCESSING_TIMEOUT: положительное конечное число секунд') from None
        path = Path(os.getenv('ASSISTANT_DB', 'data/assistant.sqlite3'))
        return cls(os.environ[required[0]], owner, os.environ[required[2]],
                   path if path.is_absolute() else ROOT / path, temperature, max_tokens, context_budget, context_reserve, processing_timeout,
                   VoiceConfig.load())
