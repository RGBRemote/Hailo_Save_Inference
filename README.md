Hailo ML Models doesnt save inference frames, rather just show the real-time detectiion and inferences. 
This repo has key changes that need to be done to save the frames at intervals. The frames will be saved in the folder containting detection.py !! Find the gstreamer_app.py in main Hailo Folder (where Models get accessed) in your system and make changes.
This is just a walkaround, not some official code. 
Hope, Hailo has no issue with the following changes.
Keeping repo open source.
