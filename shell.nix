let
  pkgs = import <nixpkgs> {
    config.allowUnfree = true;
    cudaSupport = true;
  };
  pkg-unstable = import <nixpkgs-unstable> {
    config.allowUnfree = true;
    cudaSupport = true;
  };
in
  pkgs.mkShell {
    name = "Python-CUDA-direnv";
    buildInputs =
      (
        with pkgs; [
          (python3.withPackages (python-pkgs:
            with python-pkgs; [
              isort
              black
              flake8
              pip
              numbaWithCuda
              cupy
              matplotlib
            ]))
        ]
      )
      ++ (
        with pkg-unstable; [
          linuxPackages.nvidia_x11
          julia
          cudatoolkit
          cudaPackages.cudnn
        ]
      );
    shellHook = ''
      export CUDA_PATH=${pkgs.cudatoolkit}
      export LD_LIBRARY_PATH=${pkg-unstable.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
      export EXTRA_LDFLAGS="-L/lib -L${pkg-unstable.linuxPackages.nvidia_x11}/lib"
      export EXTRA_CCFLAGS="-I/usr/include"
    '';
  }
