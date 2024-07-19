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

async def save_file(infile: UploadFile, save_dir="/tmp"):
    
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

async def wav2lip(audio_path, video_path, checkpoint, save_dir="/tmp"):

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
