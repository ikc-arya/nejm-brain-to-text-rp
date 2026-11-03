"""Stage 3 placeholder: end-to-end WER with a Whisper decoder. Not implemented yet.

The pipeline currently stops at Stage 2 (val PER). The teammate's GRU version lives in model_training/whisper/.
"""


class WhisperDecoderStage:
    """Planned contract:

    encoder   NeuralSpikeGPT (Stage 2 checkpoint), z = model.encode(x, 'b2t', session)  -> [B, T', 384]
    enc_proj  384 -> 1024 (whisper-medium d_model), masked at padding, fed as encoder_outputs to a frozen Whisper decoder
    loss      Whisper CE on the sentence + ctc_loss_weight * CTC(model.ctc_head(z)): keeps the phoneme head attached,
              so val PER is still logged and shows whether the Whisper loss erodes phoneme information
    output    beam-search decoding -> WER on the same 1426 val trials
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Stage 3 (Whisper decoder) is a placeholder; the pipeline currently ends at the Stage 2 PER.')
