{
  pkgs ? (import <nixpkgs-unstable> {
    config.allowUnfree = true;
    cudaSupport = true;
  }),
  ...
}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    git
    gitRepo
    gnupg
    autoconf
    curl
    procps
    gnumake
    util-linux
    m4
    gperf
    unzip
    libGLU
    libGL
    xorg.libXi
    xorg.libXmu
    freeglut
    xorg.libXext
    xorg.libX11
    xorg.libXv
    xorg.libXrandr
    zlib
    ncurses5
    stdenv.cc
    binutils
    cudatoolkit
    cudaPackages.cudnn
    linuxPackages.nvidia_x11
    julia
    (python3.withPackages (python-pkgs:
      with python-pkgs; [
        isort
        black
        flake8
        pip
        numbaWithCuda
        matplotlib
        #cupy
        scipy
        tqdm
      ]))
  ];
  shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
  '';
}
