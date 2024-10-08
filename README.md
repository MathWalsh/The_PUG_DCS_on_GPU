The PUG: Real-time dual comb spectroscopy (DCS) on a graphical processing unit (GPU)
---------------------------
Welcome to The PUG! Our software implements real-time corrections of a DCS interferogram stream to coherently average the interferograms, making it simpler to perform long-term experiments. It cuts down on the need for heavy post-processing and greatly reduces the amount of storage space needed.
With The PUG, you can focus more on the exciting scientific discoveries and less on the tedious signal processing!

How to use The PUG
---------------------------
1. Download the required files from the "Releases section": https://github.com/MathWalsh/The_PUG_DCS_on_GPU/releases/latest
2. Read and follow the "Instructions-and-operation-manual-for-The-PUG.pdf" file.

Hardware requirements
---------------------------
1. Windows operating system (version 10 or 11) with a minimum of 32 gB of ram (64 GB highly recommended) and 4 processing cores (4 CPU cores)
2. PCIe gage digitizer with eXpert PCIe Data Streaming Firmware  (Code tested on a CSE161G4-LR and CSE1442)
3. NVIDIA graphical processing unit (Code tested on a GeForce RTX 4090 and 4080 Super, 4070 Super, 3080 TI, 3080 laptop)
   
See instructions file for more details. 

Software requirements
---------------------------
To simply run the python GUI and the compiled C executable:
1. Gage drivers and eXpert PCIe Data Streaming Firmware
2. A Python interpreter with the necessary libraries (See instruction for the detailed list). We recommend the latest WinPython distribution (https://sourceforge.net/projects/winpython/files/)
3. Visual studio 2022 with Python development and Desktop development with C++ packages (https://visualstudio.microsoft.com/vs/)
4. CUDA Toolkit 12.3 with your windows version (10 or 11) (https://developer.nvidia.com/cuda-12-3-0-download-archive?target_os=Windows&target_arch=x86_64)
5. Matlab Runtime 2023b on windows (https://www.mathworks.com/products/compiler/matlab-runtime.html)

See instructions file for more details

To open the project and look at the C++ code, see instructions file for more details

Additional information
---------------------------
Additional information on the signal processing behind the code can be obtained from these papers : 

Self-corrected chip-based dual-comb spectrometer (https://doi.org/10.1364/OE.25.008168)

Self-Correction Limits in Dual-Comb Interferometry (https://doi.org/10.1109/JQE.2019.2918935)

Continuous real-time correction and averaging for frequency comb interferometry (https://doi.org/10.1364/OE.20.021932)

Autocorrection en temps réel pour la spectroscopie à double peigne de fréquences optiques (https://corpus.ulaval.ca/entities/publication/60da76c8-1c3d-4798-a043-05b6b48f2a30)

A paper presenting the specific details of the implementation on the GPU is coming soon. A block diagram of the algorithm is already available in the documentation folder.

Contributors
---------------------------
Mathieu Walsh : https://github.com/MathWalsh

Jérôme Genest : https://github.com/JeromeGenest

You can contact us via email : ThePugDCSonGPU@hotmail.com

License
---------------------------
Copyright (c) 2024, Mathieu Walsh, Jérôme Genest. All rights reserved.

The software is available for redistribution and use in source and binary forms, with or without modification, but strictly for non-commercial purposes only.
See License.txt for more information on the conditions.

For commercial licenses or to request custom features, please contact us by email : ThePugDCSonGPU@hotmail.com


Disclaimer
---------------------------
This is not a commercial product, there are still bugs and undetected issues in the code. To improve the quality of the code, please report any issues (https://github.com/MathWalsh/The_PUG_DCS_on_GPU/issues) encountered. If you need support to use the code, contact us via email, we will try to respond as fast as possible to get you going!
