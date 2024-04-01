{
  description = "Python CUDA direnv";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    chaotic.url = "github:chaotic-cx/nyx/nyxpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    nixpkgs,
    chaotic,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      chao = import chaotic {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      devShells.default = pkgs.mkShell {
        packages =
          (
            with pkgs; [
              julia
              cudatoolkit
              cudaPackages.cudnn
              (python3.withPackages (python-pkgs:
                with python-pkgs; [
                  isort
                  black
                  flake8
                  pip
                  numba
                  cupy
                  matplotlib
                  scipy
                ]))
            ]
          )
          ++ (
            with chao; [linuxPackages.nvidia_x11]
          );
        shellHook = ''
          export CUDA_PATH=${pkgs.cudatoolkit}
          export LD_LIBRARY_PATH=${chao.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
          export EXTRA_LDFLAGS="-L/lib -L${chao.linuxPackages.nvidia_x11}/lib"
          export EXTRA_CCFLAGS="-I/usr/include"
        '';
      };
    });
}
