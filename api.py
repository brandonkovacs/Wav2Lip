from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse
import os
import subprocess
import uuid

TMP_FOLDER = "/tmp"
WORKSPACE_FOLDER = "/workspace"

description = """
## Best Practices

### Video files
* **Format:** MP4
* **Resolution:** 1080p or 720p

### Audio files
* **Format:** WAV
* **Sample Rate:** 44,100Hz

### Checkpoints
* **wav2lip:** Highly accurate lip-sync
* **wav2lip_gan:** Slightly inferior lip-sync, but better visual quality
"""

app = FastAPI(title="wav2lip API Service",
    summary="Deepfakes as a service 😀",
    version="0.0.1",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1})

async def log(message):
    print("[+]: {}".format(str(message)))

async def save_file(infile: UploadFile, save_dir="/tmp"):
    
    # Generate a random filename to save the contents of the uploaded file
    infile_filename = f"{str(uuid.uuid4())}-{infile.filename}"
    infile_filepath = f"{save_dir}/{infile_filename}"
    
    # Write contents to disk
    try:
        contents = infile.file.read()
        with open(infile_filepath, 'wb') as f:
            f.write(contents)
    except Exception:
        return None
    finally:
        infile.file.close()

    # Return path to the saved file
    return infile_filepath

async def wav2lip(audio_path, video_path, checkpoint, output_dir="/tmp"):

    # Generate a random filename for wav2lip output file
    outfile_filename = f"{str(uuid.uuid4())}-{checkpoint}.mp4"
    outfile_path = f"{output_dir}/{outfile_filename}"

    # Run inference command
    command = 'python3 /app/inference.py --checkpoint_path /app/checkpoints/{}.pth --face {} --audio {} --outfile {}'.format(checkpoint, video_path, audio_path, outfile_path)
    results = subprocess.call(command, shell=True)
        
    # Return path to file
    return outfile_path

@app.post("/wav2lip", description=description)
async def wav2lip_inference(video_file: UploadFile = File(),
                           audio_file: UploadFile = File(),
                           checkpoint: str = Query("wav2lip_gan", enum=["wav2lip", "wav2lip_gan"])):
    
    # Upload videos to server
    video_path = await save_file(video_file, save_dir=TMP_FOLDER)
    audio_path = await save_file(audio_file, save_dir=TMP_FOLDER)

    # Output to workspace by default if it exists, otherwise TMP_DIR
    output_dir=WORKSPACE_FOLDER if os.path.exists(WORKSPACE_FOLDER) else TMP_FOLDER
    outfile = await wav2lip(audio_path=audio_path, video_path=video_path, checkpoint=checkpoint, output_dir=output_dir)

    # Print status message
    log({
        "status": "Creating wav2lip",
        "arguments": {
            "video": video_path,
            "audio": audio_path,
            "checkpoint": checkpoint,
            "output_dir": output_dir,
            "outfile": outfile
        }
    })

    # Return file
    return FileResponse(path=outfile, media_type="application/octet-stream", filename=os.path.basename(outfile))
