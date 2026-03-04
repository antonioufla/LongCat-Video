import base64
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import runpod
from huggingface_hub import snapshot_download


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
LONGCAT_VIDEO_REPO = "meituan-longcat/LongCat-Video"
LONGCAT_AVATAR_REPO = "meituan-longcat/LongCat-Video-Avatar"
MAX_LOG_LINES = 200
MODELS_READY = False
REPO_DOWNLOAD_CONFIG = {
    LONGCAT_VIDEO_REPO: {
        "local_dir": WEIGHTS_DIR / "LongCat-Video",
        "allow_patterns": [
            "tokenizer/**",
            "text_encoder/**",
            "vae/**",
            "scheduler/**",
        ],
        "required_paths": [
            "tokenizer",
            "text_encoder",
            "vae",
            "scheduler",
        ],
    },
    LONGCAT_AVATAR_REPO: {
        "local_dir": WEIGHTS_DIR / "LongCat-Video-Avatar",
        "allow_patterns": [
            "avatar_single/**",
            "chinese-wav2vec2-base/**",
            "vocal_separator/**",
        ],
        "required_paths": [
            "avatar_single",
            "chinese-wav2vec2-base",
            "vocal_separator/Kim_Vocal_2.onnx",
        ],
    },
}


def _tail_log(text: str, max_lines: int = MAX_LOG_LINES) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-max_lines:])


def _download_file(url: str, out_path: Path) -> None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=300) as response, out_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def _decode_base64_to_file(payload: str, out_path: Path) -> None:
    raw = payload.strip()
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    raw = "".join(raw.split())
    raw += "=" * (-len(raw) % 4)
    out_path.write_bytes(base64.b64decode(raw))


def _write_media(value, out_path: Path) -> None:
    if isinstance(value, dict):
        if value.get("url"):
            _download_file(value["url"], out_path)
            return
        if value.get("base64"):
            _decode_base64_to_file(value["base64"], out_path)
            return
        if value.get("data"):
            _decode_base64_to_file(value["data"], out_path)
            return
        raise ValueError("Media dict precisa conter 'url', 'base64' ou 'data'.")

    if not isinstance(value, str):
        raise ValueError("Media precisa ser string (URL/base64) ou dict.")

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        _download_file(value, out_path)
        return

    _decode_base64_to_file(value, out_path)


def _pick(input_data: dict, *keys):
    for key in keys:
        if key in input_data and input_data[key] not in (None, ""):
            return input_data[key]
    return None


def _repo_is_ready(local_dir: Path, required_paths) -> bool:
    return all((local_dir / rel_path).exists() for rel_path in required_paths)


def _ensure_repo(repo_id: str) -> None:
    config = REPO_DOWNLOAD_CONFIG[repo_id]
    local_dir = config["local_dir"]
    allow_patterns = config["allow_patterns"]
    required_paths = config["required_paths"]

    local_dir.mkdir(parents=True, exist_ok=True)
    if _repo_is_ready(local_dir, required_paths):
        return

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        token=token,
    )

    if not _repo_is_ready(local_dir, required_paths):
        raise RuntimeError(f"Download incompleto para {repo_id}.")


def ensure_models() -> None:
    global MODELS_READY
    if MODELS_READY:
        return

    _ensure_repo(LONGCAT_VIDEO_REPO)
    _ensure_repo(LONGCAT_AVATAR_REPO)
    MODELS_READY = True


def run_inference(input_data: dict) -> dict:
    ensure_models()

    prompt = input_data.get(
        "prompt",
        "A person talks naturally to the camera with clear lip sync and realistic motion.",
    )
    resolution = input_data.get("resolution", "480p")
    if resolution not in {"480p", "720p"}:
        raise ValueError("resolution deve ser '480p' ou '720p'.")

    num_segments = int(input_data.get("num_segments", 1))
    num_inference_steps = int(input_data.get("num_inference_steps", 50))
    text_guidance_scale = float(input_data.get("text_guidance_scale", 4.0))
    audio_guidance_scale = float(input_data.get("audio_guidance_scale", 4.0))

    image_value = _pick(input_data, "image", "image_base64", "image_url", "photo")
    audio_value = _pick(
        input_data,
        "wav",
        "audio",
        "audio_base64",
        "wav_base64",
        "audio_url",
        "wav_url",
    )

    if image_value is None:
        raise ValueError("Envie a imagem em 'image', 'image_base64' ou 'image_url'.")
    if audio_value is None:
        raise ValueError("Envie o audio em 'wav', 'audio', 'audio_base64' ou 'audio_url'.")

    with tempfile.TemporaryDirectory(prefix="longcat_runpod_") as temp_root:
        temp_root_path = Path(temp_root)
        media_dir = temp_root_path / "media"
        output_dir = temp_root_path / "output"
        media_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_path = media_dir / "input_image.png"
        audio_path = media_dir / "input_audio.wav"
        input_json_path = temp_root_path / "input.json"

        _write_media(image_value, image_path)
        _write_media(audio_value, audio_path)

        input_payload = {
            "prompt": prompt,
            "cond_image": str(image_path),
            "cond_audio": {"person1": str(audio_path)},
        }
        input_json_path.write_text(
            json.dumps(input_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        command = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=1",
            "run_demo_avatar_single_audio_to_video.py",
            "--context_parallel_size=1",
            f"--checkpoint_dir={WEIGHTS_DIR / 'LongCat-Video-Avatar'}",
            "--stage_1=ai2v",
            f"--input_json={input_json_path}",
            f"--output_dir={output_dir}",
            f"--resolution={resolution}",
            f"--num_segments={num_segments}",
            f"--num_inference_steps={num_inference_steps}",
            f"--text_guidance_scale={text_guidance_scale}",
            f"--audio_guidance_scale={audio_guidance_scale}",
        ]

        completed = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )
        logs = f"{completed.stdout}\n{completed.stderr}".strip()

        if completed.returncode != 0:
            raise RuntimeError(
                "Falha na inferencia LongCat.\n"
                f"Exit code: {completed.returncode}\n"
                f"Logs (tail):\n{_tail_log(logs)}"
            )

        videos = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not videos:
            raise RuntimeError(f"Nenhum mp4 encontrado. Logs (tail):\n{_tail_log(logs)}")

        final_video = videos[-1]
        video_base64 = base64.b64encode(final_video.read_bytes()).decode("utf-8")

        return {
            "video_base64": video_base64,
            "video_filename": final_video.name,
            "mime_type": "video/mp4",
            "resolution": resolution,
            "num_segments": num_segments,
            "logs_tail": _tail_log(logs),
        }


def handler(event):
    try:
        input_data = event.get("input", {})
        return run_inference(input_data)
    except Exception as exc:
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})
