# Compass Gait Walker

This repository demonstrates the passive dynamic compass gait walker demonstrated in my thesis.


## Setup
Prerequisites: python3, MuJoCo (Python API), numpy, matplotlib, scipy

Model: `bipedCompass1.xml`
<div align="center">
  <a href="CompassSnapshot1.png">
    <img src="CompassSnapshot1.png" width="300" title="Click to enlarge">
  </a>
  <br>
  <sup>Model</sup>
</div>

Script: `bipedCompassCanon3.py`



## Running
- Run `python3 bipedCompassCanon3.py` to visualize on MuJoCo and load the plots. Expected:
<div align="center">
  <a href="Compass-2steps.mov">
    <video src="Compass-2steps.mov" width="500" title="Click to enlarge">
  </a>
  <br>
  <sup>Expected output</sup>
</div>



<div align="center">
  <a href="biped_keyframes_captioned.png">
    <img src="biped_keyframes_captioned.png" width="400" title="Click to enlarge">
  </a>
  <br>
  <sup>Compass Walker</sup>
</div>


## Next Steps
Enable MuJoCo contact for dynamic walking (not just kinematic playback). Trying with `bipedCompassMJ.py`

