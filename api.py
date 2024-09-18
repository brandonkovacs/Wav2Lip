from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse
import os
import subprocess
import uuid
import pathlib

TMP_FOLDER = "/tmp"
WORKSPACE_FOLDER = "/workspace"

description = """

## Best Practices

### Video files
* **Format:** MP4
* **Resolution:** 1080p or 720p
* **Notes:** n/a

### Audio files
* **Format:** WAV
* **Sample Rate:** 44,100Hz
* **Notes:** Single speaker. No background noise

### Checkpoints
* **wav2lip:** Highly accurate lip-sync
* **wav2lip_gan:** Slightly inferior lip-sync, but better visual quality
"""

app = FastAPI(title="wav2lip web service",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1})

async def save_file(infile: UploadFile, save_dir=TMP_FOLDER):
    
    # Generate a uuid filename 
    saved_file_uuid = uuid.uuid4()
    saved_file_extension = pathlib.Path(infile.filename).suffix
    saved_file_name = f"{saved_file_uuid}{saved_file_extension}"
    saved_file_path = os.path.join(save_dir, saved_file_name)

    # Write contents to disk
    try:
        contents = infile.file.read()
        with open(saved_file_path, 'wb') as f:
            f.write(contents)
    except Exception:
        return None
    finally:
        infile.file.close()

    # Return path to the saved file
    return saved_file_path

async def wav2lip(audio_path, video_path, checkpoint, save_dir=TMP_FOLDER):

    # Generate filename for wav2lip output
    outfile_filename = f"{checkpoint}.mp4"
    outfile_path = os.path.join(save_dir, outfile_filename)

    # Run inference command
    command = 'python3 /app/inference.py --checkpoint_path /app/checkpoints/{}.pth --face {} --audio {} --outfile {}'.format(checkpoint, video_path, audio_path, outfile_path)
    results = subprocess.call(command, shell=True)
        
    # Return path to file
    return outfile_path

@app.post("/wav2lip", description=description)
async def wav2lip_inference_endpoint(video_file: UploadFile = File(),
                           audio_file: UploadFile = File(),
                           checkpoint: str = Query("wav2lip", enum=["wav2lip", "wav2lip_gan"])):
    
    # Create project folder
    project_uuid = str(uuid.uuid4())
    project_dir = os.path.join(WORKSPACE_FOLDER, project_uuid)
    os.makedirs(project_dir)

    # Upload videos to server
    video_path = await save_file(video_file, save_dir=project_dir)
    audio_path = await save_file(audio_file, save_dir=project_dir)

    # Infer wav2lip and get output file
    outfile = await wav2lip(audio_path=audio_path, video_path=video_path, checkpoint=checkpoint, save_dir=project_dir)

    # Return file
    return FileResponse(path=outfile, media_type="application/octet-stream", filename=os.path.basename(outfile))

async def esrgan_video(video_path, upscale_factor=2.0, enhance_face=True, save_dir=TMP_FOLDER):

    infile_filename = pathlib.Path(video_path).stem
    outfile_filename = f"{infile_filename}_out.mp4"
    outfile_filepath = os.path.join(save_dir, outfile_filename)

    # Run Real-ESRGAN inference command
    command = f"python3 /app/Real-ESRGAN/inference_realesrgan_video.py -i {video_path} -o {save_dir} -n RealESRGAN_x4plus -s {upscale_factor}"
    if enhance_face:
        command += " --face_enhance"
 
    results = subprocess.call(command, shell=True)
        
    # Return path to file
    return outfile_filepath

@app.post("/esrgan", description="Real-ESRGAN")
async def wav2lip_inference_endpoint(video_file: UploadFile = File(),
                           upscale_factor: str = Query("2.0", enum=["1", "1.5", "2", "2.5", "3.0", "3.5", "4.0"]),
                           enhance_face: bool = Query(True, enum=[True, False])):
    
    # Create project folder to store video file
    project_uuid = str(uuid.uuid4())
    project_dir = os.path.join(WORKSPACE_FOLDER, project_uuid)
    os.makedirs(project_dir)

    # Upload video to server
    video_path = await save_file(video_file, save_dir=project_dir)

    # Infer REAL-ESRGAN
    outfile = await esrgan_video(video_path=video_path, upscale_factor=upscale_factor, enhance_face=enhance_face, save_dir=project_dir)

    # Return file
    return FileResponse(path=outfile, media_type="application/octet-stream", filename=os.path.basename(outfile))
