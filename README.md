# m-abd-physics-godot-prototype
Implementation of concepts presented in the M-ABD paper (arXiv:2603.08079) using the Godot Engine.

Unofficial Godot implementation inspired by:
"M-ABD: Scalable, Efficient, and Robust Multi-Affine-Body Dynamics"

Paper:
https://arxiv.org/abs/2603.08079

### Overview
This project explores the implementation of concepts presented in the M-ABD paper using the Godot Engine.
Development assistance was provided by Claude Sonnet 4.6 for code discussion, debugging, and implementation support.
The goal is to study and reproduce parts of the proposed simulation framework in a practical real-time environment.

### environment
This was tested on godot 4.5.2 & godot-cpp 4.5-stable & windows 10 64bit.

### setup
1. Download godot engine 4.5.2: https://godotengine.org/download/archive/4.5.2-stable/
2. Clone this repository into the same directory where your Godot executable is located.
3. Download godot-cpp from https://github.com/godotengine/godot-cpp and put it inside your godot exe folder.
   and type "git checkout godot-4.5-stable"
4. Run build_physics.bat
5. After build is finished, run godot 4.5.2 exe and 
  import "demo/mabd-physics-test" project and change your physics engine to M-ABD.

foder structure:
```text
godot-exe
├── demo
│   ├── bin                  (generated after build)
│   └── mabd-physics-test
├── SConstruct
├── build_physics.bat
├── godot-cpp
└── src
```

### Status
Experimental work-in-progress.
A substantial portion of the M-ABD framework has been implemented, while several components described in the paper remain unfinished.
Development is ongoing, and additional features may be added in future updates if several donation is fullfilled.
This repository should not currently be considered a complete or reference implementation of the paper.
There are huge amount of Korean language comment because I need to develop this code with Korean comments. 
Also this physics code will generate huge amount of debugging message because currently this project is in progress. 
if you want to remove this, clear UtilityFunctions::print(...) for now.

### Attribution
All credit for the original research belongs to the paper authors.
Please refer to the original paper for the complete methodology and theoretical background.

```bibtex
@misc{he2026mabdscalableefficientrobust,
      title={M-ABD: Scalable, Efficient, and Robust Multi-Affine-Body Dynamics}, 
      author={Zhiyong He and Dewen Guo and Minghao Guo and Yili Zhao and Wojciech Matusik and Hao Su and Chenfanfu Jiang and Peter Yichen Chen and Yin Yang},
      year={2026},
      eprint={2603.08079},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2603.08079}, 
}

### License
Apache License 2.0

### Support Development
If you find this project useful, consider supporting its continued development.
Contributions and donations help fund additional implementation work, testing, documentation, and future features.

$5[![Donate via PayPal](https://img.shields.io/badge/Donate-PayPal-blue?style=for-the-badge&logo=paypal)](https://www.paypal.com/ncp/payment/6WVHQWKECPCTL)
$50 [![Donate via PayPal](https://img.shields.io/badge/Donate-PayPal-blue?style=for-the-badge&logo=paypal)](https://www.paypal.com/ncp/payment/6WVHQWKECPCTL)
freeamount [![Donate via PayPal](https://img.shields.io/badge/Donate-PayPal-blue?style=for-the-badge&logo=paypal)](https://www.paypal.com/ncp/payment/KAVD6P66NVASC)
