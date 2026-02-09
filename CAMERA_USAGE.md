# Camera Streaming Guide

## Overview
The ball tracking project now supports streaming from attached cameras in addition to video files.

## Usage

### Single Camera Mode

#### Using Camera with Tracking
```bash
# Use default camera (index 0)
uv run tracking --video-path 0

# Use second camera (index 1)
uv run tracking --video-path 1

# Set custom FPS (default is 120)
uv run tracking --video-path 0 --camera-fps 60

# Additional options
uv run tracking --video-path 0 --camera-fps 60 --trajectory-length 50 --show-masks --save-video
```

### Using Camera with Trajectory
```bash
# Use default camera (index 0)
uv run trajectory --video-path 0

# Use second camera (index 1)
uv run trajectory --video-path 1

# Set custom FPS (default is 120)
uv run trajectory --video-path 0 --camera-fps 30

# Additional options
uv run trajectory --video-path 0 --camera-fps 60 --show-masks
```

### Stereo Camera Mode (ELP 3D Stereo USB Camera)

For stereo cameras like the ELP 3D Stereo USB Camera, the two camera sensors typically appear as separate camera devices.

#### Using Stereo Camera with Tracking
```bash
# Use only LEFT camera (camera index 0)
uv run tracking --video-path 0 --stereo --stereo-right 1 --stereo-use left

# Use only RIGHT camera (camera index 1)
uv run tracking --video-path 0 --stereo --stereo-right 1 --stereo-use right

# Use BOTH cameras side-by-side (stitched horizontally)
uv run tracking --video-path 0 --stereo --stereo-right 1 --stereo-use both

# With custom FPS
uv run tracking --video-path 0 --stereo --stereo-right 1 --stereo-use left --camera-fps 60
```

#### Using Stereo Camera with Trajectory
```bash
# Use only LEFT camera
uv run trajectory --video-path 0 --stereo --stereo-right 1 --stereo-use left

# Use only RIGHT camera
uv run trajectory --video-path 0 --stereo --stereo-right 1 --stereo-use right

# Use BOTH cameras side-by-side
uv run trajectory --video-path 0 --stereo --stereo-right 1 --stereo-use both
```

#### Finding Your Stereo Camera Indices

To find which camera indices your stereo camera uses:
```bash
# On Linux, list video devices
ls -la /dev/video*

# Try different indices to see which shows your cameras
uv run tracking --video-path 0  # Test first camera
uv run tracking --video-path 1  # Test second camera
```

For ELP 3D Stereo cameras, typically:
- Left camera: `/dev/video0` (index 0)
- Right camera: `/dev/video1` (index 1)

## Camera Settings
The camera can be configured with the following settings:
- **Resolution**: 1280x720 (hardcoded in `video_loop.py`, can be changed to 1920x1080)
- **FPS**: Configurable via `--camera-fps` argument (default: 120)
- **Format**: MJPEG

To modify resolution, edit the camera initialization section in [video_loop.py](src/ball_tracking/video_loop.py#L82-L101).

### Common FPS Values
- `--camera-fps 30`: Standard video
- `--camera-fps 60`: Smooth video
- `--camera-fps 120`: High-speed capture (if camera supports it)

## Platform Support
- **Linux**: Uses V4L2 backend (Video4Linux2)
- **Windows**: Uses DirectShow backend
- **macOS**: Uses default backend

## Keyboard Controls
- **q**: Quit the application
- **r**: Reset tracking (for video files)

## Troubleshooting

### Camera Not Detected
If your camera is not detected:
1. Check camera index - try different values (0, 1, 2, etc.)
2. Ensure camera is not in use by another application
3. Check camera permissions on your system

### Poor Performance
If experiencing lag:
1. Lower the requested FPS in `video_loop.py`
2. Reduce the resolution
3. Adjust the `scale` parameter in tracking.py

### Camera Settings Not Applied
Some cameras may not support all requested settings. The application will use the closest supported values.
