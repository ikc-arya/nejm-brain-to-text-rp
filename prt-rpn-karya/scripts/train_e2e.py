"""Stage 3 (SpikeGPT encoder -> enc_proj 384->1024 -> Whisper decoder -> WER): placeholder, see src/stage3_whisper.py."""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from stage3_whisper import WhisperDecoderStage

if __name__ == '__main__':
    print(WhisperDecoderStage.__doc__)
    sys.exit('Stage 3 is not implemented yet; the pipeline currently ends at the Stage 2 PER.')
