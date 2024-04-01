{
  description = "Python CUDA direnv";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    chaotic.url = "github:chaotic-cx/nyx/nyxpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    nixpkgs,
    nixpkgs-unstable,
    chaotic,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
      };
      unstable = import nixpkgs-unstable {
        inherit system;
      };
      chao = import chaotic {
        inherit system;
      };
    in {
      devShells.default = pkgs.mkShell {
        packages =
          (with unstable; [
            libGLU
            libGL
            zlib
            ncurses5
            stdenv.cc
            binutils
            julia
            cudatoolkit
            cudaPackages.cudnn
          ])
          ++ (
            with pkgs; [
              (python3.withPackages (python-pkgs:
                with python-pkgs; [
                  isort
                  black
                  flake8
                  pip
                  numbaWithCuda
                  matplotlib
                  cupy
                  scipy
                  tqdm
                ]))
            ]
          )
          ++ (
            with chao; [linuxPackages.nvidia_x11]
          );

        NIXPKGS_ALLOW_UNFREE = "1";

        shellHook = ''
          export CUDA_PATH=${nixpkgs.cudatoolkit}
          export LD_LIBRARY_PATH=${chao.linuxPackages.nvidia_x11}/lib:${nixpkgs.ncurses5}/lib
          export EXTRA_LDFLAGS="-L/lib -L${chao.linuxPackages.nvidia_x11}/lib"
          export EXTRA_CCFLAGS="-I/usr/include"
        '';
      };
    });
}
