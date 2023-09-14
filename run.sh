export PYTHONPATH="${PYTHONPATH}:./extensions/sd-wav2lip-uhq/"
python run.py \
    --suno_prompt "Text to speech is a technology that allows written text to be converted into spoken words. It is commonly used in devices, software, and applications to provide an audio output for text-based content." \
    --language English \
    --gender Male \
    --video /home/feng/Downloads/Oleg_preview.mp4 \
    --pad_bottom 15 \
    --nowebui \
    --code_former_weight 1.0 \
    --face_restore_model CodeFormer
