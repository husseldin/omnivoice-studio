"""
Sonic V3 Backend for OmniVoice Studio.

Wraps the HossAI Sonic V3 inference pipeline (VAE → FPP → HFG → Vocoder)
for zero-shot Arabic TTS with voice cloning and dubbing support.

Requires:
    pip install torch torchaudio soundfile numpy pyyaml
    # And the Sonic V3 model checkpoints

Environment:
    SONIC_V3_CHECKPOINT_DIR — path to checkpoints/sonic_v3/ directory
    SONIC_V3_DEVICE — cuda | mps | cpu (auto-detected if unset)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from backend.services.tts_backend import TTSBackend

logger = logging.getLogger("omnivoice.tts.sonic_v3")


class SonicV3Backend(TTSBackend):
    """Sonic V3 adapter — Arabic-first neural TTS with prosody control."""

    NAME = "sonic_v3"
    DISPLAY_NAME = "Sonic V3 (HossAI Arabic)"
    SAMPLE_RATE = 24000

    def __init__(self, checkpoint_dir: Optional[str] = None, device: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir or os.environ.get("SONIC_V3_CHECKPOINT_DIR", ""))
        self.device = device or os.environ.get("SONIC_V3_DEVICE", "")
        self._engine: Optional[object] = None
        self._phonemizer: Optional[object] = None

    # ── TTSBackend protocol ──────────────────────────────────────────────

    @classmethod
    def is_available(cls) -> bool:
        """Check if Sonic V3 checkpoints exist and dependencies are importable."""
        try:
            import torch
            import soundfile
            ckpt_dir = os.environ.get("SONIC_V3_CHECKPOINT_DIR", "")
            if not ckpt_dir:
                return False
            p = Path(ckpt_dir)
            required = ["vae/best.pt", "fpp/best.pt", "hfg/best.pt", "vocoder/best.pt"]
            return all((p / f).exists() for f in required)
        except Exception:
            return False

    @classmethod
    def install_hint(cls) -> str:
        return (
            "Sonic V3 requires the HossAI-VoiceTTS inference pipeline and model checkpoints.\n"
            "1. Clone https://github.com/HossAI-Labs/HossAI-VoiceTTS\n"
            "2. Download checkpoints to checkpoints/sonic_v3/\n"
            "3. Set SONIC_V3_CHECKPOINT_DIR env var"
        )

    def load(self) -> None:
        """Lazy-load the Sonic V3 inference pipeline."""
        if self._engine is not None:
            return

        # Auto-detect device
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Import the pipeline (may fail if HossAI-VoiceTTS not on PYTHONPATH)
        try:
            import sys
            # Try to find the HossAI-VoiceTTS packages
            voicetts_path = Path(__file__).parent.parent.parent.parent.parent / "HossAI-VoiceTTS"
            if voicetts_path.exists():
                sys.path.insert(0, str(voicetts_path / "packages" / "sonic_v3_mvp"))
            else:
                # Fallback: assume packages are on PYTHONPATH
                pass

            from inference.pipeline import SonicV3Inference
        except ImportError as e:
            raise RuntimeError(f"Sonic V3 pipeline not found. {self.install_hint()}") from e

        ckpt = self.checkpoint_dir
        self._engine = SonicV3Inference.from_checkpoints(
            vae_ckpt=str(ckpt / "vae" / "best.pt"),
            fpp_ckpt=str(ckpt / "fpp" / "best.pt"),
            hfg_ckpt=str(ckpt / "hfg" / "best.pt"),
            vocoder_ckpt=str(ckpt / "vocoder" / "best.pt"),
            config_path=str(ckpt / ".." / ".." / "packages" / "sonic_v3_mvp" / "config.yaml"),
            device=self.device,
        )
        logger.info(f"Sonic V3 loaded on {self.device}")

    def generate(
        self,
        text: str,
        speaker_ref: Optional[str] = None,
        language: str = "ar",
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        energy_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Generate audio for ``text``.

        Args:
            text: Arabic text (or other supported language).
            speaker_ref: Path to reference audio for voice cloning (optional).
            language: Language code — ``"ar"`` for Arabic.
            speed: Speaking rate multiplier (0.5–2.0).
            pitch_shift: Pitch shift in semitones.
            energy_scale: Energy scaling.

        Returns:
            1-D float32 tensor sampled at 24 kHz.
        """
        self.load()

        # Load speaker embedding from reference if provided
        speaker_emb = None
        if speaker_ref and Path(speaker_ref).exists():
            speaker_emb = self._extract_speaker_embedding(speaker_ref)

        audio, metrics = self._engine.synthesize(
            text=text,
            speaker_emb=speaker_emb,
            speed=speed,
            pitch_shift=pitch_shift,
            energy_scale=energy_scale,
        )

        # Convert numpy array to torch tensor
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio).float()

        # Ensure 1-D
        if audio.dim() > 1:
            audio = audio.squeeze()

        return audio

    def clone_voice(self, ref_audio_path: str, text: str, **kwargs) -> torch.Tensor:
        """Zero-shot voice cloning using a reference audio clip."""
        return self.generate(text, speaker_ref=ref_audio_path, **kwargs)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _extract_speaker_embedding(self, audio_path: str) -> torch.Tensor:
        """Extract a simple speaker embedding from reference audio."""
        # For MVP: use mean spectrogram as a simple speaker embedding
        # Full implementation would use a dedicated speaker encoder
        try:
            wav, sr = torchaudio.load(audio_path)
            if sr != self.SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, self.SAMPLE_RATE)
            # Return mean energy as a simple 2-dim embedding (matches retrained FPP)
            emb = torch.tensor([wav.mean().item(), wav.std().item()], dtype=torch.float32)
            return emb.to(self.device)
        except Exception as e:
            logger.warning(f"Failed to extract speaker embedding from {audio_path}: {e}")
            return None
