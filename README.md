# Reelx  

Reelx is an advanced video editing tool that revolutionizes the transformation of horizontal videos into vertical formats. Designed to cater to the growing demand for vertical content, Reelx ensures videos are optimized for diverse platforms and audiences while maintaining high visual quality and engagement.

## Background  

- **The Rise of Vertical Video Content on Smartphones**  
  Vertical videos have become a dominant format, driven by the natural orientation of smartphones and their growing popularity on social media platforms.  

- **Enhanced Engagement and Usability**  
  Vertical videos provide a seamless viewing experience by aligning with user preferences for device-friendly content.  

- **Social Media Preference for Vertical Content**  
  Many platforms prioritize vertical videos, granting them greater visibility and reach. This trend incentivizes creators to adapt to the format to expand their audience and engagement.  

## Unique Features  

### AI-Driven Region of Interest (ROI) Adjustment  
Reelx leverages cutting-edge artificial intelligence to detect and adapt the region of interest in videos. This ensures that key visual elements remain the focal point, enhancing viewer impact and retention.  

![Reframer Demo](misc/ai-roi.gif)  

### Dynamic People Tracking  
This feature employs sophisticated algorithms to track movement and adjust focus dynamically. It ensures that individuals remain centered and prominent, even during high-motion sequences.  

![People Tracking](misc/pple_tracking.gif)  

### Tiled View for Podcast Videos  
Reelx provides a customized tiled layout for podcast videos, enabling simultaneous visibility of multiple speakers. Intelligent transitions highlight the active speaker, offering a polished and professional viewing experience.  

![Tiled view](misc/tiled_view.gif)  

### No-Loss Frame with Zoomed View  
The tool enables a zoomed-in, no-loss frame layout presented in a flexible tiled design.

![No loss frame multi position](misc/noloss_view.gif)  

### Custom Tracking Using ClassID  
Reelx introduces custom tracking capabilities that allow users to generate vertical videos focused on specific objects or areas of interest. By providing a `classID` corresponding to the target object, Reelx dynamically tracks and centers the video around the selected focus object, ensuring tailored and precise framing.  

![Custom Tracking](misc/classid.gif) 

## Getting started

## Installation

This methods of installation are as follows:

```markdown
1. Source Installation
2. Conda Environment Installation
```

### 1. Source Installation

#### Prerequisite 

1. Python version 3.12.6 or higher

2. Verify python installation

```bash
python --version
```

#### Installation Instructions

1. Clone the **Reelx** source repository usign below command

```
git clone https://github.com/Fikyo/Reelx
```

2. Traverse to the directory `src` using command `cd src`

3. install all required packages

```
pip install -r requirements.txt
```

4. execute the command below to check if the installation is successful.

```
python reelx.py -h
```

5. The command will display all available options for converting horizontal video to vertical video format.

```markdown
usage: reelx [-h] [--model_type MODEL_TYPE] [--model_verbose MODEL_VERBOSE] [--mode MODE]
                   [--output OUTPUT] [--confidence CONFIDENCE] [--smoothing SMOOTHING]
                   [--preview_vertical_video PREVIEW_VERTICAL_VIDEO]
                   [--preview_debug_player PREVIEW_DEBUG_PLAYER]
                   [--enable_tiled_frame ENABLE_TILED_FRAME] [--person_model PERSON_MODEL]
                   [--process_noloss_frame PROCESS_NOLOSS_FRAME]
                   [--noloss_tiled_position NOLOSS_TILED_POSITION]
                   input_video
```

### 2. Conda Environment Installation

#### Prerequisite 

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)

#### Installation Instructions

1. Create Conda environment for python 3.12.6 using below ommand

```
conda create -n reelx python=3.12.6
```

2. Activate the conda environment `reelx`

```
conda activate reelx
```

3. Follow the either of [#1](#1-source-installation) installation methods.
