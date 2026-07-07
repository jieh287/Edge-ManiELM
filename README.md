# Edge-ManiELM
﻿
A MATLAB implementation of **Edge-ManiELM**, a manifold-regularized extreme learning machine with edge-function superposition for small-sample soil inversion using Sentinel-2 MSI remote sensing imagery.
﻿
## Overview
﻿
This repository provides the source code of **Edge-ManiELM**, a remote sensing inversion framework designed for small-sample soil mapping in mining areas. The method combines edge-function hidden representation and spatial-spectral manifold regularization.
﻿
## Main Features
﻿
- **Edge-function hidden representation**  
Replaces fixed random hidden-layer mappings in conventional ELM with learnable mixed-order B-spline edge functions.
﻿
- **Spatial-spectral graph Laplacian regularization**  
Embeds spatial proximity and spectral similarity into the closed-form estimation of output weights.
﻿

﻿
## Requirements
﻿
The code was developed in MATLAB. The recommended environment is:
﻿
MATLAB R2021a or later
Windows 10/11
﻿
Depending on the preprocessing and visualization modules, the following MATLAB toolboxes may be required:
﻿
Statistics and Machine Learning Toolbox
Image Processing Toolbox
Mapping Toolbox

## Citation
﻿
If you use this code, please cite the associated paper:
﻿
Edge-ManiELM: Manifold Extreme Learning Machine with Edge-Function Superposition for Small-Sample SOM Spatial Inversion Using MSI Remote Sensing Imagery
﻿
## Note
﻿
This is an early public release of the Edge-ManiELM MATLAB implementation. 
The repository will be updated with cleaner modular code, example scripts, and additional documentation. 
﻿
﻿
## License
﻿
MIT License  
﻿ 
Copyright (c) 2026 jieh287  
﻿
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
﻿
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
﻿
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
﻿
﻿
﻿
﻿
