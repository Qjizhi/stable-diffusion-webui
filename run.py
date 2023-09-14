import json
from scripts.wav2lip_uhq_extend_paths import wav2lip_uhq_sys_extend
# import gradio as gr
from scripts.wav2lip.w2l import W2l
from scripts.wav2lip.wav2lip_uhq import Wav2LipUHQ
from modules.shared import state, opts
from modules import shared, options, shared_options
from modules.cmd_args import parser
from scripts.bark.tts import TTS
from modules import launch_utils
from modules import initialize

prepare_environment = launch_utils.prepare_environment

prepare_environment()
initialize.imports()
initialize.check_versions()
initialize.initialize()

speaker_json = json.load(open("extensions/sd-wav2lip-uhq/scripts/bark/speakers.json", "r"))
args = parser.parse_args()
speaker_language = args.language
speaker_gender = args.gender

speaker_list = [speaker["name"] for speaker in speaker_json if
                speaker["language"] == speaker_language and speaker["gender"] == speaker_gender]

# [TODO] Choose which to use
speaker_id = [speaker["id"] for speaker in speaker_json if speaker["name"] == speaker_list[0]][0]


def gen_audio(suno_prompt, temperature, silence, low_vram):
    global speaker_id
    if suno_prompt is None or speaker_id is None:
        return
    tts = TTS(suno_prompt, speaker_id, temperature, silence, None, low_vram)
    wav = tts.generate()
    # delete tts object to free memory
    del tts

    return wav

def generate(video, audio, checkpoint, face_restore_model, no_smooth, only_mouth, resize_factor,
                mouth_mask_dilatation, erode_face_mask, mask_blur, pad_top, pad_bottom, pad_left, pad_right,
                active_debug, code_former_weight):
    # state.begin()
    if video is None or audio is None:
        print("[ERROR] Please select a video and an audio file")
        return

    w2l = W2l(video, audio, checkpoint, no_smooth, resize_factor, pad_top, pad_bottom, pad_left,
                pad_right)
    w2l.execute()

    w2luhq = Wav2LipUHQ("./extensions/sd-wav2lip-uhq/scripts/wav2lip/temp/result.avi", face_restore_model, mouth_mask_dilatation, erode_face_mask, mask_blur,
                        only_mouth,
                        resize_factor, code_former_weight, active_debug)

    return w2luhq.execute()

def main():
    # wav2lip_uhq_sys_extend()
    audio = gen_audio(args.suno_prompt, args.temperature, args.silence, args.low_vram)
    # import IPython; IPython.embed()

    generate(args.video, audio, args.checkpoint, args.face_restore_model, args.no_smooth, args.only_mouth, args.resize_factor,
                args.mouth_mask_dilatation, args.erode_face_mask, args.mask_blur, args.pad_top, args.pad_bottom, args.pad_left, args.pad_right,
                args.active_debug, args.code_former_weight)

if __name__ == "__main__":
    main()